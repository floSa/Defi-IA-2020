"""Fine-tuning end-to-end d'un encodeur sur la tâche, objectif L1 (= la métrique du challenge).

Contrairement aux embeddings figés (scripts/encode_local.py), l'encodeur est ici entraîné *sur*
la prédiction des upvotes : ses représentations s'adaptent à la tâche au lieu d'être génériques.

Conçu pour tourner plusieurs heures sans supervision :
  - **checkpoint atomique** toutes les CKPT_EVERY étapes (modèle + optimiseur + scaler + position
    exacte dans l'époque) ; une coupure fait perdre au pire quelques minutes, jamais le run ;
  - **reprise automatique** : relancer la même commande repart du dernier checkpoint ;
  - **journal horodaté** en flux, pour qu'un superviseur externe puisse détecter un blocage.

L'entraînement exclut le holdout temporel (7 derniers jours), si bien que les prédictions
produites sur ce holdout sont des OOF valides, directement utilisables par `defia blend`.

Sorties (dans data/processed/) :
    oof/oof_<tag>.parquet, oof/test_<tag>.parquet   prédictions (pour le blend)
    train_<tag>emb.parquet, test_<tag>emb.parquet   embeddings fine-tunés réduits (pour le GBM)

Usage :
    python scripts/finetune.py --model intfloat/e5-small-v2 --tag ft_e5 --batch 256 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "src")
from defia.config import load_config
from defia.evaluation.cv import temporal_holdout_indices

CKPT_EVERY = 500          # étapes entre deux sauvegardes
LOG_EVERY = 50
SEQ_LEN = 128
EMB_COMPONENTS = 64       # cf. courbe de dimensionnalité : plateau optimal 32-64
SVD_SAMPLE = 200_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TextDS(Dataset):
    def __init__(self, texts, targets=None):
        self.texts, self.targets = texts, targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        t = self.texts[i]
        return (t, self.targets[i]) if self.targets is not None else t


class Regressor(torch.nn.Module):
    """Encodeur + pooling moyen masqué + tête linéaire scalaire."""

    def __init__(self, name: str):
        super().__init__()
        # dtype fp32 explicite : certains dépôts publient des poids fp16 que le GradScaler
        # refuse d'unscaler. L'AMP gère la fp16 au niveau des calculs, pas des paramètres.
        self.enc = AutoModel.from_pretrained(name, trust_remote_code=True, dtype=torch.float32)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def embed(self, **batch):
        h = self.enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, **batch):
        return self.head(self.embed(**batch)).squeeze(-1)


def save_ckpt(path: Path, **state) -> None:
    """Écriture atomique : un tmp puis un rename, pour qu'une coupure pendant l'écriture ne
    laisse jamais un checkpoint tronqué (sinon la reprise repartirait de zéro)."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intfloat/e5-small-v2")
    ap.add_argument("--tag", default="ft_e5")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--prefix", default="passage: ")
    ap.add_argument("--smoke", type=int, default=0,
                    help="Test de bout en bout sur N lignes (valide le chemin complet vite).")
    args = ap.parse_args()

    t_start = time.time()
    cfg = load_config("configs/default.yaml")
    processed = cfg.resolve("processed")
    oof_dir = processed / "oof"; oof_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("models") / args.tag; ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint.pt"

    log(f"modele={args.model} tag={args.tag} batch={args.batch} epochs={args.epochs}")
    log(f"gpu={torch.cuda.get_device_name(0)}")

    # --- Données : le holdout temporel est exclu de l'entraînement ---
    tr = pd.read_parquet("data/interim/train.parquet", columns=["id", "body", "ups", "created_utc"])
    if args.smoke:
        # Échantillonnage par pas régulier sur TOUTE la période : un échantillon pris en fin de
        # train tomberait entièrement dans la fenêtre de holdout (7 jours sur 24) et laisserait
        # l'ensemble d'entraînement vide.
        tr = tr.iloc[:: max(1, len(tr) // args.smoke)].reset_index(drop=True)
        log(f"MODE SMOKE : {len(tr):,} lignes réparties sur toute la période")
    fit_idx, val_idx = temporal_holdout_indices(tr["created_utc"].to_numpy(),
                                                int(cfg["cv"].get("val_days", 7)))
    texts = tr["body"].fillna("").astype(str).tolist()
    ups = tr["ups"].to_numpy(dtype="float32")
    ids = tr["id"].to_numpy()
    fit_texts = [args.prefix + texts[i] for i in fit_idx]
    fit_y = ups[fit_idx]
    log(f"fit={len(fit_idx):,} holdout={len(val_idx):,}")

    model = Regressor(args.model).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    tok = AutoTokenizer.from_pretrained(args.model)

    steps_per_epoch = len(fit_texts) // args.batch
    start_epoch, start_step, done_training = 0, 0, False
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, start_step = ck["epoch"], ck["step"]
        done_training = ck.get("done_training", False)
        log(f"REPRISE depuis checkpoint : époque {start_epoch}, étape {start_step}/{steps_per_epoch}"
            + (" (entraînement déjà terminé)" if done_training else ""))

    # --- Entraînement ---
    if not done_training:
        for epoch in range(start_epoch, args.epochs):
            # Permutation déterministe par époque : la reprise retrouve exactement le même ordre
            # et reprend à l'étape près, sans rejouer ni sauter d'exemples.
            perm = np.random.default_rng(cfg.seed + epoch).permutation(len(fit_texts))
            step0 = start_step if epoch == start_epoch else 0
            running, seen = 0.0, 0
            t_ep = time.time()
            for step in range(step0, steps_per_epoch):
                sel = perm[step * args.batch:(step + 1) * args.batch]
                bt = [fit_texts[i] for i in sel]
                by = torch.from_numpy(fit_y[sel]).cuda()
                enc = tok(bt, padding=True, truncation=True, max_length=SEQ_LEN,
                          return_tensors="pt").to("cuda")
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = torch.nn.functional.l1_loss(model(**enc), by)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                running += loss.item(); seen += 1

                if (step + 1) % LOG_EVERY == 0:
                    rate = (step + 1 - step0) * args.batch / (time.time() - t_ep)
                    eta = (steps_per_epoch - step - 1) * args.batch / max(rate, 1) / 60
                    log(f"ep{epoch} step {step+1}/{steps_per_epoch} L1={running/seen:.3f} "
                        f"{rate:.0f} ex/s ETA {eta:.0f} min")
                    running, seen = 0.0, 0
                if (step + 1) % CKPT_EVERY == 0:
                    save_ckpt(ckpt_path, model=model.state_dict(), opt=opt.state_dict(),
                              scaler=scaler.state_dict(), epoch=epoch, step=step + 1,
                              done_training=False)
            save_ckpt(ckpt_path, model=model.state_dict(), opt=opt.state_dict(),
                      scaler=scaler.state_dict(), epoch=epoch + 1, step=0, done_training=False)
            log(f"époque {epoch} terminée ({(time.time()-t_ep)/60:.0f} min)")
        save_ckpt(ckpt_path, model=model.state_dict(), opt=opt.state_dict(),
                  scaler=scaler.state_dict(), epoch=args.epochs, step=0, done_training=True)
        log(f"entraînement terminé ({(time.time()-t_start)/60:.0f} min)")

    # --- Inférence : prédictions + embeddings, sur holdout puis test ---
    model.eval()

    @torch.no_grad()
    def infer(texts_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Renvoie (prédictions, embeddings). Tri par longueur pour limiter le padding."""
        if not texts_list:
            raise ValueError("infer() appelé sur une liste vide — le découpage temporel a "
                             "probablement laissé une des partitions vide.")
        order = np.argsort([len(t) for t in texts_list], kind="stable")
        preds = np.empty(len(texts_list), dtype=np.float32)
        embs = None
        bs = args.batch * 2
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            enc = tok([texts_list[i] for i in sel], padding=True, truncation=True,
                      max_length=SEQ_LEN, return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                e = model.embed(**enc)
                p = model.head(e).squeeze(-1)
            e = e.float().cpu().numpy()
            if embs is None:
                embs = np.empty((len(texts_list), e.shape[1]), dtype=np.float32)
            preds[sel] = p.float().cpu().numpy()
            embs[sel] = e
            if (s // bs) % 200 == 0:
                log(f"  inférence {s:,}/{len(order):,}")
        return preds, embs

    log("inférence sur le holdout (OOF)...")
    val_pred, val_emb = infer([args.prefix + texts[i] for i in val_idx])
    mae = float(np.abs(val_pred - ups[val_idx]).mean())
    log(f"MAE HOLDOUT du fine-tuning seul = {mae:.4f}")
    pd.DataFrame({"id": ids[val_idx], "pred": val_pred}).to_parquet(
        oof_dir / f"oof_{args.tag}.parquet", index=False)

    log("inférence sur le test...")
    te = pd.read_parquet("data/interim/test.parquet", columns=["id", "body"])
    if args.smoke:
        te = te.head(args.smoke).reset_index(drop=True)
    te_texts = [args.prefix + t for t in te["body"].fillna("").astype(str).tolist()]
    te_pred, te_emb = infer(te_texts)
    pd.DataFrame({"id": te["id"].to_numpy(), "pred": te_pred}).to_parquet(
        oof_dir / f"test_{args.tag}.parquet", index=False)

    # Embeddings fine-tunés réduits, au format attendu par train-gbm (merge automatique)
    log("inférence sur le reste du train (pour les embeddings GBM)...")
    fit_pred, fit_emb = infer(fit_texts)
    all_emb = np.empty((len(texts), fit_emb.shape[1]), dtype=np.float32)
    all_emb[fit_idx] = fit_emb; all_emb[val_idx] = val_emb
    n_sample = min(SVD_SAMPLE, len(all_emb))
    n_comp = min(EMB_COMPONENTS, all_emb.shape[1] - 1, n_sample - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=cfg.seed).fit(
        all_emb[np.random.default_rng(cfg.seed).choice(len(all_emb), n_sample, replace=False)])
    cols = [f"emb_{i}" for i in range(n_comp)]
    for name, arr, idx_ in [("train", all_emb, ids), ("test", te_emb, te["id"].to_numpy())]:
        red = svd.transform(arr).astype(np.float32)
        df = pd.DataFrame(red, columns=cols); df.insert(0, "id", idx_)
        df.to_parquet(processed / f"{name}_{args.tag}emb.parquet", compression="zstd", index=False)
        log(f"écrit {name}_{args.tag}emb.parquet")

    json.dump({"model": args.model, "tag": args.tag, "mae_holdout": mae, "epochs": args.epochs,
               "batch": args.batch, "lr": args.lr, "minutes": (time.time() - t_start) / 60},
              open(f"reports/{args.tag}.json", "w"), indent=2)
    log(f"TERMINE — MAE holdout {mae:.4f} en {(time.time()-t_start)/60:.0f} min")


if __name__ == "__main__":
    main()
