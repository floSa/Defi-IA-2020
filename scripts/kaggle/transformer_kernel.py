"""Kernel Kaggle (GPU) — fine-tuning d'un encodeur `body -> ups` (régression).

Autonome (pas de dépendance à `defia`). Entrée : dataset Kaggle attaché avec `train.parquet`
(id, body, created_utc, ups) et `test.parquet` (id, body, created_utc).

Validation = holdout temporel (7 derniers jours du train), cohérent avec le pipeline
(cf. docs/eda_findings.md — split officiel train/test temporel, sans recouvrement).

Portée volontairement réduite pour tenir dans le quota GPU Kaggle : seqlen court (commentaires
~35 tokens), 1 epoch, sous-échantillon d'entraînement. Un seul modèle : entraîné sur le début du
train (fit), il prédit le holdout (OOF pour blending) ET le test.

Sorties (/kaggle/working/, via `kaggle kernels output`) :
    oof_transformer.parquet : id, pred (holdout)
    test_transformer.parquet : id, pred (test)
    metrics.json : MAE holdout.

Garde-fou GPU : certains GPU Kaggle (P100, sm_60) ne sont pas supportés par le torch pré-installé
(sm_70+). On teste tôt ; si incompatible, on sort avec un marqueur pour re-tenter un T4.
"""
import glob
import json
import os
import sys

# --- Fix GPU Pascal (P100, sm_60) : le torch pré-installé Kaggle ne supporte que sm_70+.
#     On installe un torch PyPI (qui, lui, supporte sm_60) et on ré-exécute le script. ---
if os.environ.get("TORCH_FIXED") != "1":
    try:
        import torch as _t
        _cap = _t.cuda.get_device_capability() if _t.cuda.is_available() else (99, 0)
    except Exception:  # noqa: BLE001
        _cap = (0, 0)
    if _cap[0] < 7:
        print(f"[kernel] GPU sm_{_cap[0]}{_cap[1]} incompatible avec le torch Kaggle "
              f"-> installation d'un torch PyPI compatible Pascal...", flush=True)
        os.system(f"{sys.executable} -m pip install -q --upgrade "
                  f"torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121")
        os.environ["TORCH_FIXED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    os.environ["TORCH_FIXED"] = "1"

import numpy as np
import pandas as pd
import torch

# DistilBERT : supporté par toute version de `transformers` (ModernBERT exige >=4.48, absent sur
# l'image Kaggle). Petit, rapide, robuste — le bon compromis pour un run GPU fiable cette nuit.
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 64
EPOCHS = 1
LR = 3e-5
VAL_DAYS = 7
SEED = 42
FIT_SUBSAMPLE = 600_000      # sous-échantillon d'entraînement (sur ~2,28M du fit)
HUBER_DELTA = 5.0
OUT_DIR = "/kaggle/working"


def check_gpu_or_exit():
    """Sort avec un marqueur clair si le GPU est absent ou incompatible (sm_60/P100)."""
    if not torch.cuda.is_available():
        print("GPU_INCOMPATIBLE: pas de CUDA disponible.")
        sys.exit(3)
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"[kernel] GPU={name} capability=sm_{cap[0]}{cap[1]}")
    try:
        x = torch.randn(64, 64, device="cuda")
        _ = (x @ x.T).sum().item()   # force un vrai kernel CUDA
    except Exception as e:  # noqa: BLE001
        print(f"GPU_INCOMPATIBLE: op CUDA échouée ({e}). GPU {name} non supporté par ce torch.")
        sys.exit(3)
    print("[kernel] GPU OK.")


def find_input_dir() -> str:
    hits = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
    if not hits:
        raise FileNotFoundError("train.parquet introuvable sous /kaggle/input")
    return os.path.dirname(hits[0])


def signed_log1p(y):
    return np.sign(y) * np.log1p(np.abs(y))


def signed_expm1(t):
    return np.sign(t) * np.expm1(np.abs(t))


def main() -> None:
    check_gpu_or_exit()
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    device = "cuda"
    input_dir = find_input_dir()
    tr = pd.read_parquet(os.path.join(input_dir, "train.parquet"),
                         columns=["id", "body", "created_utc", "ups"])
    te = pd.read_parquet(os.path.join(input_dir, "test.parquet"),
                         columns=["id", "body", "created_utc"])
    tr["body"] = tr["body"].fillna("")
    te["body"] = te["body"].fillna("")

    cutoff = int(tr["created_utc"].max()) - VAL_DAYS * 86_400
    fit_df = tr[tr["created_utc"] <= cutoff]
    val_df = tr[tr["created_utc"] > cutoff].reset_index(drop=True)
    if len(fit_df) > FIT_SUBSAMPLE:
        fit_df = fit_df.sample(n=FIT_SUBSAMPLE, random_state=SEED)
    fit_df = fit_df.reset_index(drop=True)
    print(f"[kernel] fit={len(fit_df):,} val={len(val_df):,} test={len(te):,}")

    class DS(Dataset):
        def __init__(self, texts, targets, tok):
            self.texts, self.targets, self.tok = texts, targets, tok

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = self.tok(self.texts[i] or "", truncation=True, max_length=MAX_LENGTH,
                           padding="max_length", return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            if self.targets is not None:
                item["target"] = torch.tensor(self.targets[i], dtype=torch.float32)
            return item

    class Reg(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = AutoModel.from_pretrained(MODEL_NAME)
            h = self.enc.config.hidden_size
            self.head = torch.nn.Sequential(torch.nn.Linear(h, 128), torch.nn.GELU(),
                                            torch.nn.Linear(128, 1))

        def forward(self, input_ids, attention_mask):
            out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
            return self.head(out.last_hidden_state[:, 0]).squeeze(-1)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = Reg().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = torch.nn.HuberLoss(delta=HUBER_DELTA)

    fit_loader = DataLoader(DS(fit_df["body"].tolist(), signed_log1p(fit_df["ups"].to_numpy()), tok),
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    steps = len(fit_loader) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    model.train()
    for epoch in range(EPOCHS):
        for j, batch in enumerate(fit_loader):
            ids = batch["input_ids"].to(device); att = batch["attention_mask"].to(device)
            tgt = batch["target"].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(ids, att)
                loss = loss_fn(out, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if j % 500 == 0:
                print(f"[kernel] epoch {epoch} step {j}/{len(fit_loader)} loss={loss.item():.4f}")

    @torch.no_grad()
    def predict(texts):
        model.eval()
        loader = DataLoader(DS(texts, None, tok), batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=2, pin_memory=True)
        preds = []
        for batch in loader:
            ids = batch["input_ids"].to(device); att = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast():
                preds.append(model(ids, att).float().cpu().numpy())
        return signed_expm1(np.concatenate(preds))

    val_pred = predict(val_df["body"].tolist())
    mae = float(np.mean(np.abs(val_df["ups"].to_numpy() - val_pred)))
    print(f"[kernel] HOLDOUT MAE={mae:.4f}")
    pd.DataFrame({"id": val_df["id"], "pred": val_pred}).to_parquet(
        os.path.join(OUT_DIR, "oof_transformer.parquet"), index=False)

    test_pred = predict(te["body"].tolist())
    pd.DataFrame({"id": te["id"], "pred": test_pred}).to_parquet(
        os.path.join(OUT_DIR, "test_transformer.parquet"), index=False)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({"holdout_mae": mae, "model": MODEL_NAME, "epochs": EPOCHS,
                   "fit_rows": len(fit_df)}, f, indent=2)
    print("[kernel] terminé.")


if __name__ == "__main__":
    main()
