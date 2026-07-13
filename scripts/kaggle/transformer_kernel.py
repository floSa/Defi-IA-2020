"""Kernel Kaggle (GPU) — fine-tuning d'un encodeur `body -> ups` (régression).

Autonome (pas de dépendance à `defia`). Entrée : dataset Kaggle attaché avec `train.parquet`
(id, body, created_utc, ups) et `test.parquet` (id, body, created_utc).

Validation = holdout temporel (7 derniers jours du train), cohérent avec le reste du pipeline
(cf. docs/eda_findings.md — le split officiel train/test est temporel, sans recouvrement).

Sortie (/kaggle/working/, récupérée via `kaggle kernels output`) :
    oof_transformer.parquet : id, pred (sur le holdout, pour évaluation/blending)
    test_transformer.parquet : id, pred (prédiction test finale, modèle ré-entraîné sur tout le train)
    metrics.json : MAE holdout.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 128          # commentaires courts (moyenne ~143 car., cf. docs/eda_findings.md)
BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-5
VAL_DAYS = 7
SEED = 42
HUBER_DELTA = 5.0         # perte robuste à la queue lourde de `ups`

INPUT_DIR = "/kaggle/input/defia-reddit-text"
OUT_DIR = "/kaggle/working"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CommentDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_length):
        self.texts = texts
        self.targets = targets  # None pour l'inférence pure
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i] or "", truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.targets is not None:
            item["target"] = torch.tensor(self.targets[i], dtype=torch.float32)
        return item


class RegressionHead(torch.nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # token [CLS]/BOS
        return self.head(cls).squeeze(-1)


def signed_log1p(y):
    return np.sign(y) * np.log1p(np.abs(y))


def signed_expm1(t):
    return np.sign(t) * np.expm1(np.abs(t))


def run_epoch(model, loader, optimizer=None, scheduler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    loss_fn = torch.nn.HuberLoss(delta=HUBER_DELTA)
    total_loss, preds = 0.0, []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attn = batch["attention_mask"].to(DEVICE)
        with torch.set_grad_enabled(train_mode):
            out = model(input_ids, attn)
            if "target" in batch:
                target = batch["target"].to(DEVICE)
                loss = loss_fn(out, target)
                if train_mode:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                total_loss += loss.item() * len(target)
        preds.append(out.detach().cpu().numpy())
    return total_loss / max(1, len(loader.dataset)), np.concatenate(preds)


def main() -> None:
    print(f"[kernel] device={DEVICE}, modèle={MODEL_NAME}")
    tr = pd.read_parquet(os.path.join(INPUT_DIR, "train.parquet"),
                          columns=["id", "body", "created_utc", "ups"])
    te = pd.read_parquet(os.path.join(INPUT_DIR, "test.parquet"),
                          columns=["id", "body", "created_utc"])
    tr["body"] = tr["body"].fillna("")
    te["body"] = te["body"].fillna("")

    cutoff = int(tr["created_utc"].max()) - VAL_DAYS * 86_400
    fit_df = tr[tr["created_utc"] <= cutoff].reset_index(drop=True)
    val_df = tr[tr["created_utc"] > cutoff].reset_index(drop=True)
    print(f"[kernel] fit={len(fit_df):,} val={len(val_df):,} test={len(te):,}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RegressionHead(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    fit_t = signed_log1p(fit_df["ups"].to_numpy())
    val_t = signed_log1p(val_df["ups"].to_numpy())
    fit_ds = CommentDataset(fit_df["body"].tolist(), fit_t, tokenizer, MAX_LENGTH)
    val_ds = CommentDataset(val_df["body"].tolist(), val_t, tokenizer, MAX_LENGTH)
    fit_loader = DataLoader(fit_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)

    total_steps = len(fit_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        tr_loss, _ = run_epoch(model, fit_loader, optimizer, scheduler)
        val_loss, val_pred_t = run_epoch(model, val_loader)
        val_pred = signed_expm1(val_pred_t)
        mae = float(np.mean(np.abs(val_df["ups"].to_numpy() - val_pred)))
        print(f"[kernel] epoch {epoch+1}/{EPOCHS} train_loss={tr_loss:.4f} "
              f"val_loss={val_loss:.4f} val_MAE={mae:.4f}")

    # OOF (holdout) pour blending
    pd.DataFrame({"id": val_df["id"], "pred": val_pred}).to_parquet(
        os.path.join(OUT_DIR, "oof_transformer.parquet"), index=False)

    # Ré-entraînement sur tout le train (fit+val) pour la prédiction test finale
    print("[kernel] ré-entraînement sur tout le train...")
    all_t = signed_log1p(tr["ups"].to_numpy())
    all_ds = CommentDataset(tr["body"].tolist(), all_t, tokenizer, MAX_LENGTH)
    all_loader = DataLoader(all_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model_full = RegressionHead(MODEL_NAME).to(DEVICE)
    optimizer_full = torch.optim.AdamW(model_full.parameters(), lr=LR)
    total_steps_full = len(all_loader) * EPOCHS
    scheduler_full = get_linear_schedule_with_warmup(
        optimizer_full, num_warmup_steps=int(0.06 * total_steps_full),
        num_training_steps=total_steps_full)
    for epoch in range(EPOCHS):
        loss, _ = run_epoch(model_full, all_loader, optimizer_full, scheduler_full)
        print(f"[kernel] full epoch {epoch+1}/{EPOCHS} loss={loss:.4f}")

    test_ds = CommentDataset(te["body"].tolist(), None, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2)
    _, test_pred_t = run_epoch(model_full, test_loader)
    test_pred = signed_expm1(test_pred_t)
    pd.DataFrame({"id": te["id"], "pred": test_pred}).to_parquet(
        os.path.join(OUT_DIR, "test_transformer.parquet"), index=False)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"holdout_mae": mae, "model": MODEL_NAME, "epochs": EPOCHS}, f, indent=2)
    print("[kernel] terminé.")


if __name__ == "__main__":
    main()
