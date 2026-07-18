"""Mesure le débit RÉEL d'un fine-tuning sur ce GPU, pour chiffrer le coût avant de s'engager.

Entraîne quelques dizaines de pas sur de vrais commentaires (tête de régression, perte L1 =
la métrique du challenge) et extrapole à partir du débit mesuré, pas d'une règle du pouce.
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# Balayage de la taille de batch : à 1,3 Go de VRAM pour batch=32 sur une carte de 16 Go, le
# GPU est très sous-exploité — le débit d'entraînement dépend surtout de ce paramètre.
MODELS = [("intfloat/e5-small-v2", bs) for bs in (128, 256, 512)] + \
         [("Alibaba-NLP/gte-modernbert-base", bs) for bs in (64, 128)]
SEQ_LEN = 128
N_STEPS = 40
N_TRAIN = 3_218_512
N_ALL = 4_234_970

df = pd.read_parquet("data/interim/train.parquet", columns=["body", "ups"]).head(20_000)
texts = df["body"].fillna("").astype(str).tolist()
ups = df["ups"].to_numpy(dtype="float32")


class DS(Dataset):
    def __len__(self): return len(texts)
    def __getitem__(self, i): return texts[i], ups[i]


class Regressor(torch.nn.Module):
    """Encodeur + tête linéaire sur le pooling moyen — la configuration d'un fine-tuning réel."""

    def __init__(self, name):
        super().__init__()
        # dtype float32 explicite : certains dépôts (gte-modernbert) publient des poids fp16,
        # que le GradScaler refuse ensuite d'unscaler. L'AMP gère la fp16 au niveau des calculs,
        # les paramètres eux doivent rester en fp32.
        self.enc = AutoModel.from_pretrained(name, trust_remote_code=True,
                                             dtype=torch.float32)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, **batch):
        h = self.enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


for name, bs in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = Regressor(name).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        scaler = torch.amp.GradScaler("cuda")
        loader = DataLoader(DS(), batch_size=bs, shuffle=True, drop_last=True)

        it = iter(loader)
        for warm in range(3):  # warmup hors chrono
            tb, yb = next(it)
            enc = tok(list(tb), padding=True, truncation=True, max_length=SEQ_LEN,
                      return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.l1_loss(model(**enc), yb.cuda())
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()

        torch.cuda.synchronize(); t0 = time.time(); n = 0
        for _ in range(N_STEPS):
            tb, yb = next(it)
            enc = tok(list(tb), padding=True, truncation=True, max_length=SEQ_LEN,
                      return_tensors="pt").to("cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.l1_loss(model(**enc), yb.cuda())
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
            n += len(tb)
        torch.cuda.synchronize()
        rate = n / (time.time() - t0)
        vram = torch.cuda.max_memory_allocated() / 1e9
        ep = N_TRAIN / rate / 60
        print(f"\n{name} (batch={bs})")
        print(f"  débit entraînement : {rate:,.0f} exemples/s | VRAM crête {vram:.1f} Go")
        print(f"  1 époque sur 3,22M : {ep:.0f} min")
        print(f"  2 époques + inférence sur 4,23M : {2*ep + N_ALL/6886/60:.0f} min")
        del model, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        print(f"\n{name}: ECHEC -> {type(e).__name__}: {e}")
