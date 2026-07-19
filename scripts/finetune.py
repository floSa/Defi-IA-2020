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


# Tranches d'upvotes pour l'objectif `buckets` : le binaire ups==1 n'apprend au modèle qu'à
# repérer le pic ; les tranches lui font apprendre l'INTENSITÉ de la viralité, qui est là où se
# joue la MAE (la queue). Bornes choisies sur la distribution (médiane 1, moyenne 12,7, max 6761).
BUCKETS = [1.5, 3.5, 10.5, 50.5]   # -> 5 classes : {1} · 2-3 · 4-10 · 11-50 · 51+


class Regressor(torch.nn.Module):
    """Encodeur + pooling moyen masqué + tête linéaire (scalaire, ou n_out classes)."""

    def __init__(self, name: str, n_out: int = 1):
        super().__init__()
        # dtype fp32 explicite : certains dépôts publient des poids fp16 que le GradScaler
        # refuse d'unscaler. L'AMP gère la fp16 au niveau des calculs, pas des paramètres.
        self.enc = AutoModel.from_pretrained(name, trust_remote_code=True, dtype=torch.float32)
        self.n_out = n_out
        self.head = torch.nn.Linear(self.enc.config.hidden_size, n_out)

    def embed(self, **batch):
        h = self.enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, **batch):
        out = self.head(self.embed(**batch))
        return out if self.n_out > 1 else out.squeeze(-1)


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
    ap.add_argument("--objective", default="l1", choices=["l1", "cls", "buckets"],
                    help="l1 = régression directe sur ups (s'effondre sur la médiane : "
                         "52%% des ups valent 1, et la constante optimale en L1 EST la "
                         "médiane) ; cls = classification ups==1, équilibrée 52/48 donc "
                         "insensible à ce collapsus, et c'est le signal qu'exploite le "
                         "modèle deux étages.")
    ap.add_argument("--crossfit-k", type=int, default=0,
                    help="Nombre de plis pour la generation hors-echantillon des features. "
                         "Sans cela, le modele predit les lignes qu'il a vues a l'entrainement "
                         "et ses features sont optimistes de 0,04 a 0,08 d'AUC sur ces lignes : "
                         "le GBM leur fait alors trop confiance.")
    ap.add_argument("--crossfit-fold", type=int, default=-1,
                    help="Indice du pli a tenir hors entrainement (0..k-1).")
    args = ap.parse_args()

    t_start = time.time()
    cfg = load_config("configs/default.yaml")
    processed = cfg.resolve("processed")
    oof_dir = processed / "oof"; oof_dir.mkdir(parents=True, exist_ok=True)
    # Le pli fait partie de l'identite du run : sans cela le pli 1 reprend le checkpoint du
    # pli 0, le croit termine, saute l'entrainement et reutilise SON modele — ce qui annule
    # exactement la separation que la validation croisee doit garantir.
    run_name = args.tag if args.crossfit_fold < 0 else f"{args.tag}_fold{args.crossfit_fold}"
    ckpt_dir = Path("models") / run_name; ckpt_dir.mkdir(parents=True, exist_ok=True)
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
    # --- Validation croisée : ce pli est retiré de l'entraînement pour que ses prédictions
    # soient hors échantillon (et donc comparables à celles du holdout et du test). ---
    held_idx = np.array([], dtype=int)
    if args.crossfit_k > 1:
        assert 0 <= args.crossfit_fold < args.crossfit_k, "--crossfit-fold hors bornes"
        rng = np.random.default_rng(cfg.seed)
        assign = rng.integers(0, args.crossfit_k, size=len(fit_idx))
        held_idx = fit_idx[assign == args.crossfit_fold]
        fit_idx = fit_idx[assign != args.crossfit_fold]
        log(f"CROSSFIT pli {args.crossfit_fold}/{args.crossfit_k} : "
            f"{len(fit_idx):,} lignes d'entrainement, {len(held_idx):,} tenues a l'ecart")

    fit_texts = [args.prefix + texts[i] for i in fit_idx]
    # En classification, la cible est l'indicatrice ups==1 (et non ups lui-même).
    if args.objective == "cls":
        fit_y = (ups[fit_idx] == 1).astype("float32")
    elif args.objective == "buckets":
        fit_y = np.digitize(ups[fit_idx], BUCKETS).astype("int64")
    else:
        fit_y = ups[fit_idx]
    log(f"fit={len(fit_idx):,} holdout={len(val_idx):,} objectif={args.objective}")
    if args.objective == "cls":
        log(f"part de ups==1 dans le fit : {fit_y.mean():.1%} (equilibre -> pas de collapsus)")
    elif args.objective == "buckets":
        rep = np.bincount(fit_y, minlength=len(BUCKETS) + 1) / len(fit_y)
        log("repartition des tranches {1}/2-3/4-10/11-50/51+ : "
            + " ".join(f"{r:.1%}" for r in rep))

    n_out = len(BUCKETS) + 1 if args.objective == "buckets" else 1
    model = Regressor(args.model, n_out=n_out).cuda()
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
                    out = model(**enc)
                    if args.objective == "cls":
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, by)
                    elif args.objective == "buckets":
                        loss = torch.nn.functional.cross_entropy(out, by)
                    else:
                        loss = torch.nn.functional.l1_loss(out, by)
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
        preds = np.empty((len(texts_list), n_out) if n_out > 1 else len(texts_list),
                         dtype=np.float32)
        embs = None
        bs = args.batch * 2
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            enc = tok([texts_list[i] for i in sel], padding=True, truncation=True,
                      max_length=SEQ_LEN, return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                e = model.embed(**enc)
                p = model.head(e)
                if n_out == 1:
                    p = p.squeeze(-1)
            e = e.float().cpu().numpy()
            if embs is None:
                embs = np.empty((len(texts_list), e.shape[1]), dtype=np.float32)
            preds[sel] = p.float().cpu().numpy()
            embs[sel] = e
            if (s // bs) % 200 == 0:
                log(f"  inférence {s:,}/{len(order):,}")
        return preds, embs

    # --- Mode validation croisée : on ne produit que les PRÉDICTIONS (les embeddings
    # fine-tunés se sont montrés perdants, seule la sortie du modèle nous intéresse). ---
    if args.crossfit_k > 1:
        cf_dir = processed / "crossfit"; cf_dir.mkdir(parents=True, exist_ok=True)
        te = pd.read_parquet("data/interim/test.parquet", columns=["id", "body"])
        parts = {
            "held": (ids[held_idx], [args.prefix + texts[i] for i in held_idx]),
            "val": (ids[val_idx], [args.prefix + texts[i] for i in val_idx]),
            "test": (te["id"].to_numpy(),
                     [args.prefix + t for t in te["body"].fillna("").astype(str)]),
        }
        for part, (pids, ptexts) in parts.items():
            log(f"inférence {part} ({len(ptexts):,})...")
            pred, _ = infer(ptexts)
            df = pd.DataFrame({"id": pids})
            if n_out > 1:
                z = pred - pred.max(1, keepdims=True)
                pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
                for k in range(n_out):
                    df[f"emb_pb{k}"] = pr[:, k].astype(np.float32)
                centers = np.array([1.0, 2.5, 7.0, 30.0, 120.0], dtype=np.float32)
                df["emb_exp"] = (pr * centers).sum(1).astype(np.float32)
            else:
                df["emb_p1"] = (1.0 / (1.0 + np.exp(-pred))).astype(np.float32)
            out = cf_dir / f"{args.tag}_fold{args.crossfit_fold}_{part}.parquet"
            df.to_parquet(out, compression="zstd", index=False)
            log(f"  écrit {out.name}")
        json.dump({"model": args.model, "tag": args.tag, "objective": args.objective,
                   "crossfit_k": args.crossfit_k, "crossfit_fold": args.crossfit_fold,
                   "minutes": (time.time() - t_start) / 60},
                  open(f"reports/{args.tag}_fold{args.crossfit_fold}.json", "w"), indent=2)
        log(f"TERMINE (pli {args.crossfit_fold}) en {(time.time()-t_start)/60:.0f} min")
        return

    log("inférence sur le holdout...")
    val_pred, val_emb = infer([args.prefix + texts[i] for i in val_idx])
    if args.objective == "buckets":
        from sklearn.metrics import roc_auc_score
        yb = np.digitize(ups[val_idx], BUCKETS)
        # Score comparable au binaire : AUC de « n'est PAS dans la tranche {1} », obtenue en
        # sommant les probabilités des tranches supérieures.
        pr = np.exp(val_pred - val_pred.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        score = float(roc_auc_score((yb > 0).astype(int), 1.0 - pr[:, 0]))
        acc = float((pr.argmax(1) == yb).mean())
        log(f"AUC HOLDOUT (ups!=1, via les tranches) = {score:.4f} | exactitude 5 classes = {acc:.3f}")
        log(f"  a comparer a 0.6057 pour l'objectif binaire")
    elif args.objective == "cls":
        from sklearn.metrics import roc_auc_score
        score = float(roc_auc_score((ups[val_idx] == 1).astype(int), val_pred))
        log(f"AUC HOLDOUT (texte seul, ups==1) = {score:.4f}")
        # La sortie est une probabilité, pas une prédiction d'ups : elle n'a rien à faire
        # dans le blend (qui mélange des prédictions de la cible). Elle part en feature.
    else:
        score = float(np.abs(val_pred - ups[val_idx]).mean())
        log(f"MAE HOLDOUT du fine-tuning seul = {score:.4f}")
        pd.DataFrame({"id": ids[val_idx], "pred": val_pred}).to_parquet(
            oof_dir / f"oof_{args.tag}.parquet", index=False)

    log("inférence sur le test...")
    te = pd.read_parquet("data/interim/test.parquet", columns=["id", "body"])
    if args.smoke:
        te = te.head(args.smoke).reset_index(drop=True)
    te_texts = [args.prefix + t for t in te["body"].fillna("").astype(str).tolist()]
    te_pred, te_emb = infer(te_texts)
    if args.objective == "l1":
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
    # En classification, la probabilité prédite est elle-même une feature de premier ordre
    # pour le GBM (c'est l'étage A du modèle deux étages, appris sur le texte seul).
    all_p = np.empty((len(texts), n_out) if n_out > 1 else len(texts), dtype=np.float32)
    all_p[fit_idx] = fit_pred; all_p[val_idx] = val_pred
    probs = {"train": all_p, "test": te_pred}


    def softmax(z):
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    for name, arr, idx_ in [("train", all_emb, ids), ("test", te_emb, te["id"].to_numpy())]:
        red = svd.transform(arr).astype(np.float32)
        df = pd.DataFrame(red, columns=cols); df.insert(0, "id", idx_)
        if args.objective == "cls":
            df["emb_p1"] = 1.0 / (1.0 + np.exp(-probs[name]))  # logit -> probabilité
        elif args.objective == "buckets":
            pr = softmax(probs[name])
            for k in range(n_out):
                df[f"emb_pb{k}"] = pr[:, k].astype(np.float32)
            # Espérance de l'ups sous la loi prédite : une seule feature qui résume l'intensité
            # attendue, souvent plus exploitable par le GBM que les probabilités brutes.
            centers = np.array([1.0, 2.5, 7.0, 30.0, 120.0], dtype=np.float32)
            df["emb_exp"] = (pr * centers).sum(1).astype(np.float32)
        df.to_parquet(processed / f"{name}_{args.tag}emb.parquet", compression="zstd", index=False)
        log(f"écrit {name}_{args.tag}emb.parquet")

    json.dump({"model": args.model, "tag": args.tag, "objective": args.objective,
               ("auc_holdout" if args.objective in ("cls", "buckets") else "mae_holdout"): score,
               "epochs": args.epochs,
               "batch": args.batch, "lr": args.lr, "minutes": (time.time() - t_start) / 60},
              open(f"reports/{args.tag}.json", "w"), indent=2)
    log(f"TERMINE — score holdout {score:.4f} en {(time.time()-t_start)/60:.0f} min")


if __name__ == "__main__":
    main()
