"""Point d'entrée CLI du pipeline Défi IA 2020.

Chaque sous-commande correspond à une étape du plan (docs/plan.md §6) et est appelée par le
Makefile. La CLI charge la config YAML puis délègue aux modules du package. Les étapes de
modélisation lèvent ``NotImplementedError`` tant que le plan n'est pas validé — c'est
volontaire : le squelette est en place, l'implémentation suit la validation.

Usage :
    python -m defia.cli <étape> --config configs/default.yaml
"""
from __future__ import annotations

import click

from defia.config import load_config


@click.group()
def main() -> None:
    """Défi IA 2020 — prédiction des upvotes Reddit (MAE)."""


def _cfg(config: str):
    cfg = load_config(config)
    click.echo(f"[defia] config: {cfg.path}")
    return cfg


@main.command()
@click.option("--config", default="configs/default.yaml")
def data(config: str) -> None:
    """Raw CSV -> data/interim/{train,test}.parquet."""
    cfg = _cfg(config)
    from defia.data.load import build_train_test

    counts = build_train_test(cfg.resolve("raw_csv"), cfg.resolve("interim"))
    click.echo(f"[data] écrit train={counts['train']:,} test={counts['test']:,} "
               f"-> {cfg.resolve('interim')}")


@main.command()
@click.option("--config", default="configs/default.yaml")
def baseline(config: str) -> None:
    """Baselines MAE (médiane globale + médianes par groupe) en HOLDOUT TEMPOREL.

    Le split officiel étant temporel (train 1-24 mai, test 25-31 mai), on valide sur les 7
    derniers jours du train : c'est le juge de paix. On rapporte aussi le GroupKFold à titre de
    diagnostic (optimiste sur ce problème).
    """
    import json

    import numpy as np
    import pandas as pd

    from defia.data.load import load_split
    from defia.evaluation.cv import group_kfold_indices, temporal_holdout_indices
    from defia.evaluation.metrics import mae
    from defia.models.baseline import predict_group_median

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    target = cfg["data"]["target"]
    group = cfg["cv"]["group"]
    val_days = int(cfg["cv"].get("val_days", 7))

    click.echo("[baseline] chargement du train...")
    tr = load_split(interim, "train", columns=["id", target, group, "created_utc", "author"])
    tr["hour"] = (tr["created_utc"] // 3600) % 24
    y = tr[target].to_numpy(dtype=float)
    n = len(tr)
    click.echo(f"[baseline] {n:,} lignes ; médiane={np.median(y):g} moyenne={y.mean():.3f}")

    def eval_baselines(fit_df, val_df) -> dict[str, float]:
        yv = val_df[target].to_numpy(dtype=float)
        med = float(fit_df[target].median())
        preds = {
            "global_median": np.full(len(val_df), med),
            "median_by_hour": predict_group_median(fit_df, val_df, "hour", target),
            "median_by_author": predict_group_median(fit_df, val_df, "author", target),
            "const_1": np.ones(len(val_df)),
            "const_mean": np.full(len(val_df), fit_df[target].mean()),
        }
        return {k: mae(yv, p) for k, p in preds.items()}

    # --- Juge de paix : holdout temporel (7 derniers jours) ---
    fit_idx, val_idx = temporal_holdout_indices(tr["created_utc"].to_numpy(), val_days)
    temporal = eval_baselines(tr.iloc[fit_idx], tr.iloc[val_idx])
    click.echo(f"\n[baseline] MAE — HOLDOUT TEMPOREL ({val_days} derniers jours, "
               f"fit={len(fit_idx):,} / val={len(val_idx):,}) :")
    for name, val in sorted(temporal.items(), key=lambda kv: kv[1]):
        click.echo(f"   {name:20s} {val:.4f}")

    # --- Diagnostic : GroupKFold thread (optimiste) ---
    oof = {k: np.zeros(n) for k in ("global_median", "median_by_hour", "median_by_author")}
    for tri, vai in group_kfold_indices(tr[group].to_numpy(), 5, cfg.seed):
        f, v = tr.iloc[tri], tr.iloc[vai]
        oof["global_median"][vai] = float(f[target].median())
        oof["median_by_hour"][vai] = predict_group_median(f, v, "hour", target)
        oof["median_by_author"][vai] = predict_group_median(f, v, "author", target)
    groupkf = {k: mae(y, p) for k, p in oof.items()}
    click.echo("\n[baseline] MAE — GroupKFold thread (diagnostic, optimiste) :")
    for name, val in sorted(groupkf.items(), key=lambda kv: kv[1]):
        click.echo(f"   {name:20s} {val:.4f}")

    reports = cfg.resolve("reports"); reports.mkdir(parents=True, exist_ok=True)
    (reports / "baseline_mae.json").write_text(
        json.dumps({"temporal_holdout": temporal, "group_kfold": groupkf}, indent=2),
        encoding="utf-8")

    # Première soumission : médiane globale du train pour tout le test (format id,predicted)
    te = load_split(interim, "test", columns=["id"])
    best_const = float(np.median(y))
    sub = pd.DataFrame({"id": te["id"], "predicted": best_const})
    subs = cfg.resolve("submissions"); subs.mkdir(parents=True, exist_ok=True)
    out = subs / "submission_baseline_median.csv"
    sub.to_csv(out, index=False)
    click.echo(f"\n[baseline] submission : {out} ({len(sub):,} lignes, predicted={best_const:g})")


@main.command()
@click.option("--config", default="configs/default.yaml")
@click.option("--chunk", default=300_000, help="Taille de chunk pour la stylométrie (mémoire).")
@click.option("--sentiment/--no-sentiment", default=None, help="Force VADER on/off (surcharge config).")
def features(config: str, chunk: int, sentiment) -> None:
    """Construit les matrices de features (réseau + texte) -> data/processed/{split}_features.parquet.

    Streaming mémoire-léger (WSL ~7 Go) : structural calculé une fois sur l'union train+test
    (l'ordre concat(train,test) est préservé => alignement positionnel avec ``struct``), puis
    stylométrie par chunks avec écriture parquet incrémentale.
    """
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from defia.data.load import load_split
    from defia.features.structural import build_structural_features
    from defia.features.stylometric import build_stylometric_features

    import gc

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    processed = cfg.resolve("processed"); processed.mkdir(parents=True, exist_ok=True)
    with_sent = bool(cfg["features"].get("sentiment", True)) if sentiment is None else sentiment
    target = cfg["data"]["target"]
    struct_path = processed / "_struct.parquet"

    # 1) Structural sur l'UNION (transductif, sans label). union = concat(train, test) => train d'abord.
    #    Écrit sur disque puis libéré : son pic mémoire ne chevauche pas celui de la stylométrie.
    scols = ["id", "created_utc", "link_id", "name", "author", "parent_id"]
    n_train = load_split(interim, "train", ["id"]).shape[0]
    if not struct_path.exists():
        click.echo("[features] structural (union train+test)...")
        tr = load_split(interim, "train", scols)
        te = load_split(interim, "test", scols)
        union = pd.concat([tr, te], ignore_index=True)
        del tr, te; gc.collect()
        struct = build_structural_features(union)
        struct["created_utc"] = union["created_utc"].to_numpy()
        del union; gc.collect()
        struct.to_parquet(struct_path, compression="zstd", index=False)
        click.echo(f"[features] structural écrit ({struct.shape[1]} col x {len(struct):,}) -> {struct_path}")
        del struct; gc.collect()
    else:
        click.echo(f"[features] structural déjà présent ({struct_path}), réutilisé.")

    # 2) Stylométrie par split, en chunks, alignée positionnellement sur struct (relu du disque)
    struct = pd.read_parquet(struct_path)
    bases = {"train": 0, "test": n_train}
    for split in ("train", "test"):
        base = bases[split]
        ups = (load_split(interim, "train", [target])[target].to_numpy()
               if split == "train" else None)
        pf = pq.ParquetFile(interim / f"{split}.parquet")
        out = processed / f"{split}_features.parquet"
        writer = None; offset = 0
        click.echo(f"[features] stylométrie {split} (sentiment={with_sent}, chunk={chunk:,})...")
        for batch in pf.iter_batches(batch_size=chunk, columns=["id", "body"]):
            b = batch.to_pandas()
            m = len(b)
            sty = build_stylometric_features(b, with_sentiment=with_sent)
            ssub = struct.iloc[base + offset: base + offset + m].reset_index(drop=True)
            assert (ssub["id"].to_numpy() == sty["id"].to_numpy()).all(), "désalignement id!"
            feat = pd.concat([ssub, sty.drop(columns="id")], axis=1)
            if ups is not None:
                feat[target] = ups[offset: offset + m]
            tbl = pa.Table.from_pandas(feat, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            offset += m
            click.echo(f"    {split}: {offset:,} lignes")
        writer.close()
        click.echo(f"[features] écrit {out} ({offset:,} lignes)")


@main.command("author-encoding")
@click.option("--config", default="configs/default.yaml")
@click.option("--smoothing", default=20.0)
def author_encoding(config: str, smoothing: float) -> None:
    """Target encoding auteur temporellement propre -> data/processed/{split}_author_enc.parquet."""
    from defia.data.load import load_split
    from defia.features.author_target import build_author_history_features

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    processed = cfg.resolve("processed"); processed.mkdir(parents=True, exist_ok=True)
    target = cfg["data"]["target"]

    click.echo("[author-enc] chargement...")
    tr = load_split(interim, "train", ["id", "author", "created_utc", target])
    te = load_split(interim, "test", ["id", "author", "created_utc"])
    tr_feat, te_feat = build_author_history_features(tr, te, target=target, smoothing=smoothing)
    del tr, te

    tr_feat.to_parquet(processed / "train_author_enc.parquet", compression="zstd", index=False)
    te_feat.to_parquet(processed / "test_author_enc.parquet", compression="zstd", index=False)
    click.echo(f"[author-enc] écrit train={len(tr_feat):,} test={len(te_feat):,} "
               f"(smoothing={smoothing}); mean(author_hist_mean) train="
               f"{tr_feat['author_hist_mean'].mean():.3f}")


@main.command("context-features")
@click.option("--config", default="configs/default.yaml")
def context_features(config: str) -> None:
    """Features réseau avancées de contexte (parent, vélocité thread, dynamique auteur intra-fil)."""
    import pandas as pd

    from defia.data.load import load_split
    from defia.features.context import build_context_features

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    processed = cfg.resolve("processed"); processed.mkdir(parents=True, exist_ok=True)
    scols = ["id", "created_utc", "link_id", "name", "author", "parent_id"]
    tr = load_split(interim, "train", scols); n_train = len(tr)
    te = load_split(interim, "test", scols)
    union = pd.concat([tr, te], ignore_index=True); del tr, te
    ctx = build_context_features(union); del union
    ctx.iloc[:n_train].to_parquet(processed / "train_context.parquet", compression="zstd", index=False)
    ctx.iloc[n_train:].to_parquet(processed / "test_context.parquet", compression="zstd", index=False)
    click.echo(f"[context] écrit {ctx.shape[1]-1} features x {len(ctx):,} lignes")


@main.command("author-dynamics")
@click.option("--config", default="configs/default.yaml")
@click.option("--smoothing", default=20.0)
def author_dynamics(config: str, smoothing: float) -> None:
    """Réputation d'auteur enrichie temporelle (mean/std/max/viral/down) -> *_author_dyn.parquet."""
    from defia.data.load import load_split
    from defia.features.author_target import build_author_dynamics

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    processed = cfg.resolve("processed"); processed.mkdir(parents=True, exist_ok=True)
    target = cfg["data"]["target"]
    tr = load_split(interim, "train", ["id", "author", "created_utc", target])
    te = load_split(interim, "test", ["id", "author", "created_utc"])
    d_tr, d_te = build_author_dynamics(tr, te, target=target, smoothing=smoothing)
    d_tr.to_parquet(processed / "train_author_dyn.parquet", compression="zstd", index=False)
    d_te.to_parquet(processed / "test_author_dyn.parquet", compression="zstd", index=False)
    click.echo(f"[author-dyn] écrit {d_tr.shape[1]-1} features x train={len(d_tr):,} test={len(d_te):,}")


@main.command("interactions")
@click.option("--config", default="configs/default.yaml")
def interactions_cmd(config: str) -> None:
    """Interactions non linéaires + encodage cyclique heure/jour -> *_interactions.parquet."""
    import pandas as pd

    from defia.features.interactions import build_interactions

    cfg = _cfg(config)
    processed = cfg.resolve("processed")
    for split in ("train", "test"):
        base = pd.read_parquet(processed / f"{split}_features.parquet", columns=["id", "hour", "dow", "n_chars", "depth"])
        ctx = pd.read_parquet(processed / f"{split}_context.parquet")
        adyn = pd.read_parquet(processed / f"{split}_author_dyn.parquet")
        out = build_interactions(base, ctx, adyn)
        out.to_parquet(processed / f"{split}_interactions.parquet", compression="zstd", index=False)
        click.echo(f"[interactions] {split}: {out.shape[1]-1} features x {len(out):,}")


@main.command("parent-enc")
@click.option("--config", default="configs/default.yaml")
def parent_enc(config: str) -> None:
    """Encodage du contexte parent (parent_ups + réputation de l'auteur du parent)."""
    from defia.data.load import load_split
    from defia.features.parent_enc import build_parent_encoding

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    processed = cfg.resolve("processed"); processed.mkdir(parents=True, exist_ok=True)
    target = cfg["data"]["target"]
    cols = ["id", "name", "author", "parent_id", "created_utc"]
    tr = load_split(interim, "train", cols + [target])
    te = load_split(interim, "test", cols)
    p_tr, p_te = build_parent_encoding(tr, te, target=target)
    p_tr.to_parquet(processed / "train_parentenc.parquet", compression="zstd", index=False)
    p_te.to_parquet(processed / "test_parentenc.parquet", compression="zstd", index=False)
    click.echo(f"[parent-enc] écrit {p_tr.shape[1]-1} features x train={len(p_tr):,} test={len(p_te):,}")


@main.command("kaggle-export")
@click.option("--config", default="configs/default.yaml")
@click.option("--out", default="scripts/kaggle/dataset", help="Dossier de sortie pour le dataset Kaggle.")
def kaggle_export(config: str, out: str) -> None:
    """Exporte un sous-ensemble léger (id, body, created_utc[, ups]) pour upload en dataset Kaggle."""
    from pathlib import Path

    from defia.config import REPO_ROOT
    from defia.data.load import load_split

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    out_path = Path(out)
    out_dir = out_path if out_path.is_absolute() else REPO_ROOT / out_path
    out_dir.mkdir(parents=True, exist_ok=True)
    target = cfg["data"]["target"]

    tr = load_split(interim, "train", ["id", "body", "created_utc", target])
    te = load_split(interim, "test", ["id", "body", "created_utc"])
    tr.to_parquet(out_dir / "train.parquet", compression="zstd", index=False)
    te.to_parquet(out_dir / "test.parquet", compression="zstd", index=False)
    click.echo(f"[kaggle-export] écrit {out_dir}/train.parquet ({len(tr):,}) "
               f"et test.parquet ({len(te):,})")


@main.command("tfidf-features")
@click.option("--config", default="configs/default.yaml")
@click.option("--chunk", default=150_000)
@click.option("--sample-size", default=300_000, help="Taille d'échantillon train pour ajuster la SVD.")
def tfidf_features(config: str, chunk: int, sample_size: int) -> None:
    """TF-IDF (hashing) + SVD -> data/processed/{split}_tfidf.parquet (features denses tfidf_svd_*)."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from defia.data.load import unescape_body
    from defia.features.text import fit_svd_on_sample, transform_tfidf_svd

    cfg = _cfg(config)
    interim = cfg.resolve("interim")
    processed = cfg.resolve("processed"); processed.mkdir(parents=True, exist_ok=True)
    tcfg = cfg["features"]["tfidf"]
    n_comp = int(tcfg.get("svd_components", 128))
    n_feat = int(tcfg.get("n_features", 2 ** 18))
    ngram = tuple(tcfg.get("word_ngram", [1, 2]))

    click.echo(f"[tfidf] ajustement SVD sur échantillon train (n={sample_size:,})...")
    sample = pd.read_parquet(interim / "train.parquet", columns=["body"]).sample(
        n=min(sample_size, 3_000_000), random_state=cfg.seed)
    hv, svd = fit_svd_on_sample(unescape_body(sample["body"]), n_feat, ngram, n_comp, cfg.seed)
    del sample
    click.echo(f"[tfidf] SVD ajustée : {n_comp} composantes, variance expliquée cumulée="
               f"{svd.explained_variance_ratio_.sum():.3f}")

    cols = [f"tfidf_svd_{i}" for i in range(n_comp)]
    for split in ("train", "test"):
        pf = pq.ParquetFile(interim / f"{split}.parquet")
        out = processed / f"{split}_tfidf.parquet"
        writer = None
        click.echo(f"[tfidf] transform {split} (chunk={chunk:,})...")
        offset = 0
        for batch in pf.iter_batches(batch_size=chunk, columns=["id", "body"]):
            b = batch.to_pandas()
            vecs = transform_tfidf_svd(unescape_body(b["body"]), hv, svd)
            feat = pd.DataFrame(vecs, columns=cols)
            feat.insert(0, "id", b["id"].to_numpy())
            tbl = pa.Table.from_pandas(feat, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            offset += len(b)
            click.echo(f"    {split}: {offset:,} lignes")
        writer.close()
        click.echo(f"[tfidf] écrit {out} ({offset:,} lignes)")


@main.command("train-gbm")
@click.option("--config", default="configs/default.yaml")
@click.option("--log-target/--no-log-target", default=None, help="Entraîner sur log1p(ups).")
@click.option("--objective", default=None, help="mae | huber | quantile (surcharge la config).")
@click.option("--tag", default="gbm", help="Nom de la soumission/artefacts.")
@click.option("--eval-only", is_flag=True, help="Mesure la MAE holdout puis s'arrête (itération rapide).")
@click.option("--sample", default=0, type=int, help="Sous-échantillonne le fit à N lignes (itération rapide).")
@click.option("--lr", default=0.0, type=float, help="Override learning_rate (0 = config).")
@click.option("--rounds", default=0, type=int, help="Override n_estimators (0 = config).")
def train_gbm(config: str, log_target, objective, tag: str, eval_only: bool, sample: int,
              lr: float, rounds: int) -> None:
    """Entraîne LightGBM (objectif MAE) — holdout temporel + soumission test."""
    import json

    import numpy as np
    import pandas as pd

    from defia.evaluation.cv import temporal_holdout_indices
    from defia.evaluation.metrics import mae, mae_report
    from defia.models.gbm import feature_columns, fit_predict, lgb_params, train_full_predict

    cfg = _cfg(config)
    processed = cfg.resolve("processed")
    val_days = int(cfg["cv"].get("val_days", 7))
    gbm_cfg = dict(cfg["model"]["gbm"])
    if objective:
        gbm_cfg["objective"] = objective
    if lr:
        gbm_cfg["learning_rate"] = lr
    if rounds:
        gbm_cfg["n_estimators"] = rounds
    lt = bool(cfg.get("target_transform") == "log1p") if log_target is None else log_target

    click.echo("[gbm] chargement des features...")
    tr = pd.read_parquet(processed / "train_features.parquet")
    te = pd.read_parquet(processed / "test_features.parquet")
    ae_tr, ae_te = processed / "train_author_enc.parquet", processed / "test_author_enc.parquet"
    if ae_tr.exists() and ae_te.exists():
        tr = tr.merge(pd.read_parquet(ae_tr), on="id", how="left")
        te = te.merge(pd.read_parquet(ae_te), on="id", how="left")
        click.echo("[gbm] + author_hist_mean / author_hist_count_log (target encoding auteur)")
    tf_tr, tf_te = processed / "train_tfidf.parquet", processed / "test_tfidf.parquet"
    if tf_tr.exists() and tf_te.exists():
        tr = tr.merge(pd.read_parquet(tf_tr), on="id", how="left")
        te = te.merge(pd.read_parquet(tf_te), on="id", how="left")
        click.echo("[gbm] + tfidf_svd_* (TF-IDF hashing + SVD)")
    em_tr, em_te = processed / "train_emb.parquet", processed / "test_emb.parquet"
    if em_tr.exists() and em_te.exists():
        tr = tr.merge(pd.read_parquet(em_tr), on="id", how="left")
        te = te.merge(pd.read_parquet(em_te), on="id", how="left")
        click.echo("[gbm] + emb_* (embeddings de phrase, Kaggle GPU)")
    for name, label in [("context", "contexte (parent, vélocité, dynamique intra-fil)"),
                        ("author_dyn", "dynamique auteur (mean/std/max/viral/down)"),
                        ("parentenc", "encodage cible du parent (réputation auteur parent)"),
                        ("interactions", "interactions + encodage cyclique heure/jour"),
                        ("graph", "embeddings de graphe node2vec")]:
        ptr, pte = processed / f"train_{name}.parquet", processed / f"test_{name}.parquet"
        if ptr.exists() and pte.exists():
            tr = tr.merge(pd.read_parquet(ptr), on="id", how="left")
            te = te.merge(pd.read_parquet(pte), on="id", how="left")
            click.echo(f"[gbm] + {name}_* ({label})")
    cols = feature_columns(tr)
    click.echo(f"[gbm] {len(cols)} features, train={len(tr):,}, test={len(te):,}, "
               f"objective={gbm_cfg['objective']}, log_target={lt}")

    # --- Holdout temporel (juge de paix) ---
    fit_idx, val_idx = temporal_holdout_indices(tr["created_utc"].to_numpy(), val_days)
    if sample and sample < len(fit_idx):  # sous-échantillonne le fit pour itérer vite (val complet)
        rng = np.random.default_rng(cfg.seed)
        fit_idx = np.sort(rng.choice(fit_idx, size=sample, replace=False))
        click.echo(f"[gbm] fit sous-échantillonné à {sample:,} lignes (itération rapide)")
    df_fit, df_val = tr.iloc[fit_idx].copy(), tr.iloc[val_idx].copy()
    if eval_only:  # en éval, on n'a plus besoin du gros DataFrame -> libère la mémoire
        del tr; import gc; gc.collect()
    params = lgb_params(gbm_cfg, cfg.seed)
    val_pred, _, model, best = fit_predict(
        df_fit, df_val, None, cols, "ups", params,
        early_stopping_rounds=int(gbm_cfg.get("early_stopping_rounds", 200)), log_target=lt)
    rep = mae_report(df_val["ups"].to_numpy(), val_pred, baseline_value=1.0)
    click.echo(f"\n[gbm] HOLDOUT TEMPOREL — MAE={rep['mae']:.4f} "
               f"(baseline médiane {rep['mae_baseline']:.4f}, gain {rep['gain_rel']:+.1%}), "
               f"best_iter={best}")

    # Importances (top 15)
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=cols).sort_values(ascending=False)
    click.echo("[gbm] top features (gain):\n" + imp.head(15).to_string())

    if eval_only:
        click.echo(f"[gbm] (eval-only) MAE holdout={rep['mae']:.4f} — arrêt sans soumission.")
        return

    # --- Soumission : ré-entraînement sur tout le train, prédiction test ---
    test_pred, _ = train_full_predict(tr, te, cols, "ups", params, best, log_target=lt)
    test_pred = np.clip(test_pred, -50, None)  # bornes douces (ups peut être négatif)
    subs = cfg.resolve("submissions"); subs.mkdir(parents=True, exist_ok=True)
    out = subs / f"submission_{tag}.csv"
    pd.DataFrame({"id": te["id"], "predicted": test_pred}).to_csv(out, index=False)
    click.echo(f"[gbm] submission -> {out}")

    # --- OOF (holdout) + test predictions pour le blending (Milestone E) ---
    oof_dir = processed / "oof"; oof_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df_val["id"].to_numpy(), "pred": val_pred}).to_parquet(
        oof_dir / f"oof_{tag}.parquet", index=False)
    pd.DataFrame({"id": te["id"].to_numpy(), "pred": test_pred}).to_parquet(
        oof_dir / f"test_{tag}.parquet", index=False)
    click.echo(f"[gbm] OOF + test preds -> {oof_dir}/(oof|test)_{tag}.parquet")

    reports = cfg.resolve("reports"); reports.mkdir(parents=True, exist_ok=True)
    (reports / f"gbm_{tag}.json").write_text(json.dumps({
        "objective": gbm_cfg["objective"], "log_target": lt, "best_iter": int(best),
        "holdout": rep, "top_features": imp.head(20).to_dict(),
    }, indent=2), encoding="utf-8")


@main.command()
@click.option("--config", default="configs/default.yaml")
def embeddings(config: str) -> None:
    """Extrait les embeddings de phrase (étape C, GPU)."""
    _cfg(config)
    raise SystemExit("Étape 'embeddings' à implémenter après validation du plan (GPU).")


@main.command("train-transformer")
@click.option("--config", default="configs/default.yaml")
def train_transformer(config: str) -> None:
    """Fine-tune l'encodeur body->ups (étape D, GPU)."""
    _cfg(config)
    raise SystemExit("Étape 'train-transformer' à implémenter après validation du plan (GPU).")


@main.command()
@click.option("--config", default="configs/default.yaml")
@click.option("--tag", default="blend", help="Nom de la soumission finale.")
def blend(config: str, tag: str) -> None:
    """Blending OOF des modèles (poids optimisant la MAE sur le holdout) -> soumission finale."""
    import json

    import numpy as np
    import pandas as pd

    from defia.data.load import load_split
    from defia.evaluation.cv import temporal_holdout_indices
    from defia.evaluation.metrics import mae
    from defia.models.blend import optimize_weights_mae

    cfg = _cfg(config)
    processed = cfg.resolve("processed")
    oof_dir = processed / "oof"
    target = cfg["data"]["target"]
    val_days = int(cfg["cv"].get("val_days", 7))

    # Vérité holdout (7 derniers jours du train) : id -> ups
    tr = load_split(cfg.resolve("interim"), "train", ["id", target, "created_utc"])
    _, val_idx = temporal_holdout_indices(tr["created_utc"].to_numpy(), val_days)
    truth = tr.iloc[val_idx][["id", target]].rename(columns={target: "y"})

    # Découvre les modèles : paires oof_<name>.parquet / test_<name>.parquet
    models = []
    for p in sorted(oof_dir.glob("oof_*.parquet")):
        name = p.stem[len("oof_"):]
        tpath = oof_dir / f"test_{name}.parquet"
        if tpath.exists():
            models.append(name)
    if not models:
        raise SystemExit(f"Aucun modèle OOF trouvé dans {oof_dir} (lance train-gbm / rapatrie le transformer).")
    click.echo(f"[blend] {len(models)} modèles : {', '.join(models)}")

    # Matrice OOF alignée sur la vérité holdout
    base = truth.copy()
    for name in models:
        o = pd.read_parquet(oof_dir / f"oof_{name}.parquet").rename(columns={"pred": name})
        base = base.merge(o, on="id", how="left")
    base = base.dropna(subset=models)
    y = base["y"].to_numpy(dtype=float)
    X = base[models].to_numpy(dtype=float)

    singles = {name: mae(y, base[name].to_numpy()) for name in models}
    w = optimize_weights_mae(X, y)
    blend_mae = mae(y, X @ w)

    click.echo("[blend] MAE holdout par modèle :")
    for name, m in sorted(singles.items(), key=lambda kv: kv[1]):
        click.echo(f"   {name:26s} {m:.4f}   poids={w[models.index(name)]:.3f}")
    best_single = min(singles.values())
    click.echo(f"[blend] ★ BLEND MAE={blend_mae:.4f}  (meilleur seul {best_single:.4f}, "
               f"gain {(best_single-blend_mae)/best_single:+.2%})")

    # Application des poids au test -> soumission
    te_ids = pd.read_parquet(oof_dir / f"test_{models[0]}.parquet")[["id"]]
    tmat = te_ids.copy()
    for name in models:
        t = pd.read_parquet(oof_dir / f"test_{name}.parquet").rename(columns={"pred": name})
        tmat = tmat.merge(t, on="id", how="left")
    test_pred = np.clip(tmat[models].to_numpy(dtype=float) @ w, -50, None)

    subs = cfg.resolve("submissions"); subs.mkdir(parents=True, exist_ok=True)
    out = subs / f"submission_{tag}.csv"
    pd.DataFrame({"id": tmat["id"], "predicted": test_pred}).to_csv(out, index=False)
    reports = cfg.resolve("reports"); reports.mkdir(parents=True, exist_ok=True)
    (reports / f"blend_{tag}.json").write_text(json.dumps({
        "models": models, "weights": dict(zip(models, w.tolist())),
        "singles_mae": singles, "blend_mae": blend_mae, "best_single": best_single,
    }, indent=2), encoding="utf-8")
    click.echo(f"[blend] soumission finale -> {out}")


@main.command()
@click.option("--config", default="configs/default.yaml")
def submit(config: str) -> None:
    """Écrit submissions/submission.csv (id,predicted)."""
    _cfg(config)
    raise SystemExit("Étape 'submit' à implémenter après validation du plan.")


if __name__ == "__main__":
    main()
