"""
train_pubmedbert_mesh_tl_v9.py
================================
PubMedBERT + Spurious Direction Subtraction (v9)
for GZSL MeSH Classification.

Architecture
------------
BERT (frozen)
    → CLS  h  [batch, 768]
    → SpuriousTranslationLayer  (v9)
        Phase 1 (warm-up epochs): z = h          (translation OFF, gamma=0)
        Phase 2 (remaining epochs):
            gate  s = top_k_gate(sigmoid(W·h + b), top_k)   [batch, num_seen]
            proj  = h @ V_seen.T                              [batch, num_seen]
            raw_signal = (s * proj) @ V_seen                  [batch, 768]
            z = h - softplus(log_gamma) * raw_signal
    → HybridClassifier
        seen:   seen_head  Linear(768 → num_seen)      (trainable)
        unseen: query_adapter(z) → L2 → cosine / tau   (trainable)

Key improvements over v6
-------------------------
* Warm-up phase   : translation is OFF for --warmup_epochs epochs so the
                    classifier has calibrated weights before the gate activates.
                    Solves the circular bootstrap failure in embryology.
* Learnable gamma : log_gamma is a scalar nn.Parameter (init log(0.1)).
                    softplus(log_gamma) ≥ 0 always — no clipping needed.
                    The model learns how aggressively to debias.
* Top-k gate      : only the top-k sigmoid scores per document contribute to
                    the translation. Limits correlated-direction accumulation.
                    --top_k 0 disables sparsification (all scores used).
* V uses seen directions only: gate is over seen labels; we never gate on
                    unseen labels (their gate score is always 0 in v6 too,
                    so this is an explicit clean design rather than a silent
                    coincidence).
* Per-domain HPs  : all hyperparameters (lr, epochs, warmup_epochs, top_k,
                    gamma_init, batch_size) can differ per domain via CLI.




Notes on per-domain tuning
--------------------------
Start with the defaults above (same across domains) and tune:
  --warmup_epochs  : how many epochs before gate activates (try 3, 5, 8)
  --top_k          : 0 = all gates active; try avg_labels_per_doc for domain
                     embryology ~12, immunology ~12, neurology ~9
  --lr             : 1e-3 is the best found from v8b sweep; adjust per domain
  --gamma_init     : log-space init for learnable gamma (default -2.3 ≈ 0.1)
                     lower = more conservative debiasing at start
"""

import argparse
import json
import logging
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PUBMEDBERT_CKPT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


# ===========================================================================
# 1. Label extraction
# ===========================================================================

def _extract_entry_fields(
    entry: dict,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    normalised: Dict[str, str] = {}
    for k, v in entry.items():
        base = re.sub(r'_\d+$', '', k)
        normalised[base] = v
    return (
        normalised.get("mesh_heading") or None,
        normalised.get("tree_numbers") or None,
        normalised.get("unique_id")    or None,
    )


def extract_labels_from_item(item: dict, label_mode: str) -> List[str]:
    enhanced = item.get("meshMajorEnhanced", []) or []
    labels: List[str] = []
    for entry in enhanced:
        heading, tree_nums_raw, unique_id = _extract_entry_fields(entry)
        if label_mode == "unique_id":
            if unique_id:
                labels.append(unique_id)
        elif label_mode == "heading":
            if heading:
                labels.append(heading)
        elif label_mode == "tree_numbers":
            if tree_nums_raw:
                for t in tree_nums_raw.split(","):
                    t = t.strip()
                    if t:
                        labels.append(t)
        elif label_mode == "combined":
            if not unique_id:
                continue
            if tree_nums_raw:
                for t in tree_nums_raw.split(","):
                    t = t.strip()
                    if t:
                        labels.append(f"{unique_id}||{t}")
            else:
                labels.append(f"{unique_id}||NONE")
    seen_set, deduped = set(), []
    for lbl in labels:
        if lbl not in seen_set:
            seen_set.add(lbl)
            deduped.append(lbl)
    return deduped


# ===========================================================================
# 2. Data loading
# ===========================================================================

def load_json_items(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    if isinstance(raw, dict):
        if all(isinstance(v, dict) for v in raw.values()):
            items = []
            for pmid_key, record in raw.items():
                if "pmid" not in record:
                    record = dict(record)
                    record["pmid"] = str(pmid_key)
                items.append(record)
            logger.info(f"Detected dict-of-dicts -> {len(items)} records.")
            return items
        list_values = [(k, v) for k, v in raw.items() if isinstance(v, list)]
        if len(list_values) == 1:
            return [e for e in list_values[0][1] if isinstance(e, dict)]
        raise ValueError(f"Ambiguous JSON structure. Keys: {list(raw.keys())[:10]}")
    raise ValueError(f"Unexpected JSON root type: {type(raw).__name__}")


# ===========================================================================
# 3. Label vocabulary
# ===========================================================================

def build_label_maps(
    train_items: List[dict],
    label_mode: str,
    unseen_codes: List[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns:
        seen_map   : labels in training data   → indices 0..num_seen-1
        unseen_map : held-out codes            → indices 0..num_unseen-1
    Full label_map for test = seen first, unseen after.
    """
    from collections import Counter
    counter: Counter = Counter()
    for item in train_items:
        for lbl in extract_labels_from_item(item, label_mode):
            counter[lbl] += 1

    unseen_set    = set(unseen_codes)
    seen_labels   = sorted(lbl for lbl in counter if lbl not in unseen_set)
    unseen_labels = sorted(code for code in unseen_codes if counter.get(code, 0) == 0)

    seen_map   = {lbl: idx for idx, lbl in enumerate(seen_labels)}
    unseen_map = {lbl: idx for idx, lbl in enumerate(unseen_labels)}

    logger.info(
        f"Label maps: {len(seen_map)} seen, {len(unseen_map)} unseen "
        f"(total {len(seen_map) + len(unseen_map)})"
    )
    return seen_map, unseen_map


def compute_pos_weight(
    items: List[dict],
    label_map: Dict[str, int],
    label_mode: str,
) -> torch.Tensor:
    n      = len(items)
    counts = np.zeros(len(label_map), dtype=np.float32)
    for item in items:
        for lbl in extract_labels_from_item(item, label_mode):
            if lbl in label_map:
                counts[label_map[lbl]] += 1
    counts     = np.maximum(counts, 1.0)
    pos_weight = np.clip((n - counts) / counts, 0.1, 100.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


# ===========================================================================
# 4. Dataset
# ===========================================================================

class MeshDataset(Dataset):
    def __init__(
        self,
        items: List[dict],
        tokenizer,
        label_map: Dict[str, int],
        label_mode: str,
        max_len: int,
        split_name: str = "",
    ):
        self.items      = items
        self.tokenizer  = tokenizer
        self.label_map  = label_map
        self.label_mode = label_mode
        self.max_len    = max_len

        self.label_vecs: List[np.ndarray] = []
        dropped = 0
        for item in items:
            vec = np.zeros(len(label_map), dtype=np.float32)
            for lbl in extract_labels_from_item(item, label_mode):
                if lbl in label_map:
                    vec[label_map[lbl]] = 1.0
                else:
                    dropped += 1
            self.label_vecs.append(vec)

        if dropped:
            logger.info(
                f"[{split_name}] {dropped} label occurrences not in "
                f"label_map (dropped). Expected for val/test."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item     = self.items[idx]
        title    = (item.get("title") or "").strip()
        abstract = (item.get("abstractText") or "").strip()
        text     = (
            title + " [SEP] " + abstract if title and abstract
            else title or abstract
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.label_vecs[idx], dtype=torch.float32),
        }


# ===========================================================================
# 5. Spurious direction estimation  (orthogonal, same as v6)
# ===========================================================================

def load_abstract_mesh(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"pmid": str, "mesh_id": str})
    required = {"pmid", "mesh_id", "is_spurious"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"abstract_mesh CSV missing columns: {missing}. Found: {list(df.columns)}"
        )
    df["is_spurious"] = df["is_spurious"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    logger.info(
        f"abstract_mesh: {len(df)} rows, {df['pmid'].nunique()} PMIDs, "
        f"{df['is_spurious'].sum()} spurious (pmid,label) pairs"
    )
    return df


@torch.no_grad()
def collect_cls_embeddings(
    bert_model: nn.Module,
    tokenizer,
    items: List[dict],
    label_map: Dict[str, int],
    label_mode: str,
    max_len: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, List[str]]:
    bert_model.eval()
    ds     = MeshDataset(items, tokenizer, label_map, label_mode, max_len, "cls_collect")
    loader = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, num_workers=4)

    all_cls  = []
    all_pmid = [str(item.get("pmid", i)) for i, item in enumerate(items)]

    for batch in tqdm(loader, desc="Collecting CLS embeddings"):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        out  = bert_model(input_ids=ids, attention_mask=mask)
        cls  = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_cls.append(cls)

    return np.concatenate(all_cls, axis=0), all_pmid


def estimate_spurious_directions(
    embeddings: np.ndarray,
    pmids: List[str],
    label_map: Dict[str, int],
    abstract_mesh_df: pd.DataFrame,
) -> np.ndarray:
    """
    Per-label orthogonal spurious direction estimation.

    For each label l with both spurious and non-spurious training docs:
        v_raw  = mean(spurious_embeddings) - mean(clean_embeddings)
        e_gen  = mean_clean / ||mean_clean||        (genuine semantic axis)
        v_orth = v_raw - (v_raw · e_gen) * e_gen    (project out genuine)
        v[l]   = v_orth / ||v_orth||

    Labels with only spurious docs, only clean docs, or absent from the KG
    get v[l] = 0 (no subtraction performed).

    Returns: directions [num_labels, 768]
    """
    num_labels = len(label_map)
    hidden     = embeddings.shape[1]
    directions = np.zeros((num_labels, hidden), dtype=np.float32)

    pmid_to_idx: Dict[str, int] = {p: i for i, p in enumerate(pmids)}

    mesh_records: Dict[str, Dict[str, bool]] = {}
    for _, row in abstract_mesh_df.iterrows():
        mid = str(row["mesh_id"])
        pid = str(row["pmid"])
        if mid not in mesh_records:
            mesh_records[mid] = {}
        mesh_records[mid][pid] = bool(row["is_spurious"])

    stats = {"both": 0, "spur_only": 0, "clean_only": 0, "absent": 0}

    for label, idx in tqdm(label_map.items(), desc="Estimating spurious directions"):
        records = mesh_records.get(label, {})
        if not records:
            stats["absent"] += 1
            continue

        spur_embs, clean_embs = [], []
        for pmid, is_spur in records.items():
            emb_idx = pmid_to_idx.get(pmid)
            if emb_idx is None:
                continue
            (spur_embs if is_spur else clean_embs).append(embeddings[emb_idx])

        if spur_embs and clean_embs:
            mean_spur  = np.mean(spur_embs,  axis=0)
            mean_clean = np.mean(clean_embs, axis=0)
            v_raw      = mean_spur - mean_clean
            e_genuine  = mean_clean / (np.linalg.norm(mean_clean) + 1e-8)
            v_orth     = v_raw - np.dot(v_raw, e_genuine) * e_genuine
            norm       = np.linalg.norm(v_orth)
            if norm > 1e-8:
                directions[idx] = v_orth / norm
            stats["both"] += 1
        elif spur_embs and not clean_embs:
            stats["spur_only"] += 1
        else:
            stats["clean_only"] += 1

    logger.info(
        f"Direction estimation complete:"
        f"\n  Both sides (active directions) : {stats['both']}"
        f"\n  Always spurious  (v=0)         : {stats['spur_only']}"
        f"\n  Always clean     (v=0)         : {stats['clean_only']}"
        f"\n  Absent from KG   (v=0)         : {stats['absent']}"
        f"\n  Total labels                   : {num_labels}"
    )
    return directions


# ===========================================================================
# 6. Label embedding builder  (for unseen zero-shot scoring)
# ===========================================================================

def build_label_embeddings(
    label_map: Dict[str, int],
    nodes_csv_path: str,
    bert_model: nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    nodes_df     = pd.read_csv(nodes_csv_path, dtype={"mesh_id": str})
    mesh_to_name = dict(zip(nodes_df["mesh_id"], nodes_df["pref_name"].fillna("")))

    num_labels   = len(label_map)
    hidden       = bert_model.config.hidden_size
    E            = torch.zeros(num_labels, hidden, dtype=torch.float32)
    idx_to_label = {v: k for k, v in label_map.items()}
    all_indices  = sorted(idx_to_label.keys())

    texts, idxs, missing = [], [], 0
    for idx in all_indices:
        label = idx_to_label[idx]
        name  = mesh_to_name.get(label, "")
        if not name:
            missing += 1
            name = label
        texts.append(name)
        idxs.append(idx)

    if missing:
        logger.warning(f"{missing} labels missing from nodes.csv — using MeSH ID as text.")

    logger.info(f"Encoding {len(texts)} label names with PubMedBERT ...")
    bert_model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Label embeddings"):
            batch_texts = texts[start: start + batch_size]
            batch_idxs  = idxs[start:  start + batch_size]
            enc = tokenizer(
                batch_texts, max_length=64,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            out = bert_model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            cls = out.last_hidden_state[:, 0, :].cpu().float()
            for i, gidx in enumerate(batch_idxs):
                E[gidx] = cls[i]

    logger.info(f"Label embedding matrix: {E.shape}")
    return E


# ===========================================================================
# 7. Model
# ===========================================================================

def top_k_gate(s: torch.Tensor, k: int) -> torch.Tensor:
    """
    Sparsify gate scores: keep only the top-k values per document,
    zero out the rest.  k=0 disables sparsification (returns s unchanged).

    Args:
        s : [batch, num_labels]  — sigmoid gate scores in [0, 1]
        k : number of active gates per document (0 = all)
    Returns:
        s_sparse : [batch, num_labels]  — same shape, mostly zeros
    """
    if k <= 0:
        return s
    # torch.topk on the last dim; values below threshold become 0
    batch, L = s.shape
    k_clamped = min(k, L)
    topk_vals, topk_idx = torch.topk(s, k_clamped, dim=1)
    s_sparse = torch.zeros_like(s)
    s_sparse.scatter_(1, topk_idx, topk_vals)
    return s_sparse


class SpuriousTranslationLayerV9(nn.Module):
    """
    v9 Translation Layer.

    z = h - softplus(log_gamma) * (s_sparse * (h @ V.T)) @ V

    where:
        V          : [num_seen, 768]  fixed buffer of unit spurious directions
                     (only seen labels; unseen are never gated)
        log_gamma  : learnable scalar, init = gamma_init (log-space)
                     softplus ensures gamma >= 0 always
        s          : [batch, num_seen]  sigmoid gate from classifier weights
        s_sparse   : top-k sparsified version of s (or s if top_k=0)

    The translation is a no-op when:
        (a) warmup_active=True  (gate is disabled externally by setting s=0)
        (b) V[l] = 0            (label has no valid spurious direction)
        (c) s[i,l] ≈ 0         (model is not predicting label l for doc i)
    """

    def __init__(self, num_seen: int, hidden_size: int = 768, gamma_init: float = -2.3):
        super().__init__()
        # V covers seen labels only — explicit and clean
        self.register_buffer("V", torch.zeros(num_seen, hidden_size))
        # log_gamma: learnable, softplus(log_gamma) = gamma ≥ 0
        self.log_gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.num_seen  = num_seen

    def set_directions(self, directions: np.ndarray):
        """directions: [num_seen, 768]  — only the seen-label rows."""
        assert directions.shape[0] == self.num_seen, (
            f"Expected {self.num_seen} directions, got {directions.shape[0]}"
        )
        self.V.copy_(torch.from_numpy(directions).float())
        n_active = (self.V.norm(dim=1) > 1e-8).sum().item()
        logger.info(
            f"SpuriousTranslationLayerV9: {n_active}/{self.num_seen} "
            f"active (non-zero) directions."
        )

    def forward(
        self,
        h: torch.Tensor,          # [batch, 768]
        s: torch.Tensor,          # [batch, num_seen]  sigmoid gate scores
        top_k: int = 0,
        active: bool = True,      # False during warm-up → z = h
    ) -> torch.Tensor:
        if not active:
            return h

        gamma        = F.softplus(self.log_gamma)          # scalar ≥ 0
        s_sparse     = top_k_gate(s, top_k)                # [batch, num_seen]
        proj         = h @ self.V.T                        # [batch, num_seen]
        raw_signal   = (s_sparse * proj) @ self.V          # [batch, 768]
        return h - gamma * raw_signal


class HybridClassifier(nn.Module):
    """
    Seen labels  → Linear head  (trainable)
    Unseen labels → cosine(query_adapter(z), E_unseen) / tau  (trainable)
    """

    def __init__(self, num_seen: int, num_unseen: int, hidden_size: int = 768):
        super().__init__()
        self.num_seen   = num_seen
        self.num_unseen = num_unseen

        self.seen_head = nn.Linear(hidden_size, num_seen)

        self.query_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
        )
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
        self.register_buffer("E_unseen", torch.zeros(num_unseen, hidden_size))

        nn.init.eye_(self.query_adapter[0].weight)
        nn.init.zeros_(self.query_adapter[0].bias)

    def set_unseen_embeddings(self, E: torch.Tensor):
        self.E_unseen.copy_(F.normalize(E.float(), dim=1))
        logger.info(f"Unseen embeddings set: {tuple(self.E_unseen.shape)}, L2-normalised.")

    def forward_seen(self, z: torch.Tensor) -> torch.Tensor:
        return self.seen_head(z)

    def forward_all(self, z: torch.Tensor) -> torch.Tensor:
        seen_logits   = self.seen_head(z)
        q             = F.normalize(self.query_adapter(z), dim=1)
        tau           = self.log_tau.exp().clamp(min=1e-4)
        unseen_logits = (q @ self.E_unseen.T) / tau
        return torch.cat([seen_logits, unseen_logits], dim=1)


class PubMedBERTWithTLV9(nn.Module):
    """
    BERT (frozen) → SpuriousTranslationLayerV9 → HybridClassifier

    Phases controlled by self.warmup_active:
        warmup_active=True  → translation is bypassed (z = h)
        warmup_active=False → translation is live with learned gamma + top-k gate

    train_mode=True  → forward returns seen-only logits  [batch, num_seen]
    train_mode=False → forward returns all logits        [batch, num_seen + num_unseen]
    """

    def __init__(
        self,
        bert_model,
        num_seen: int,
        num_unseen: int,
        gamma_init: float = -2.3,
    ):
        super().__init__()
        hidden           = bert_model.config.hidden_size
        self.bert        = bert_model
        self.translation = SpuriousTranslationLayerV9(num_seen, hidden, gamma_init)
        self.classifier  = HybridClassifier(num_seen, num_unseen, hidden)
        self.num_seen    = num_seen
        self.num_unseen  = num_unseen
        self.train_mode  = True
        self.warmup_active = True   # set to False after warm-up epochs
        self.top_k         = 0      # set from args

        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # ── Step 1: frozen BERT ──────────────────────────────────────────
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            h   = out.last_hidden_state[:, 0, :].float().detach()  # [batch, 768]

        # ── Step 2: gate scores (seen labels only) ───────────────────────
        # Always compute from detached classifier weights so grad_fn for
        # seen_head is only created in step 3 (where it needs to be).
        with torch.no_grad():
            W_gate = self.classifier.seen_head.weight.detach()   # [num_seen, 768]
            b_gate = self.classifier.seen_head.bias.detach()     # [num_seen]
            s      = torch.sigmoid(h @ W_gate.T + b_gate)        # [batch, num_seen]

        # ── Step 3: translation (no-op during warm-up) ───────────────────
        # log_gamma participates in the forward pass here (gets gradients).
        z = self.translation(
            h, s,
            top_k=self.top_k,
            active=not self.warmup_active,
        )

        # ── Step 4: classify ─────────────────────────────────────────────
        if self.train_mode:
            return self.classifier.forward_seen(z)
        else:
            return self.classifier.forward_all(z)


# ===========================================================================
# 8. Metrics
# ===========================================================================

def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    n, p = y_true.shape[0], 0.0
    for i in range(n):
        top_k = np.argsort(scores[i])[::-1][:k]
        p    += y_true[i][top_k].sum() / k
    return p / n


def ndcg_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    n, ndcg = y_true.shape[0], 0.0
    for i in range(n):
        top_k = np.argsort(scores[i])[::-1][:k]
        rel   = y_true[i][top_k]
        dcg   = sum(rel[j] / math.log2(j + 2) for j in range(len(rel)))
        ideal = sum(
            1.0 / math.log2(j + 2)
            for j in range(min(int(y_true[i].sum()), k))
        )
        ndcg += dcg / ideal if ideal > 0 else 0.0
    return ndcg / n


def f1_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[float, float]:
    preds = (scores >= threshold).astype(np.float32)
    tp    = (preds * y_true).sum()
    fp    = (preds * (1 - y_true)).sum()
    fn    = ((1 - preds) * y_true).sum()
    p     = tp / (tp + fp + 1e-8)
    r     = tp / (tp + fn + 1e-8)
    micro_f1  = 2 * p * r / (p + r + 1e-8)
    label_f1s = []
    for j in range(y_true.shape[1]):
        tp_j = (preds[:, j] * y_true[:, j]).sum()
        fp_j = (preds[:, j] * (1 - y_true[:, j])).sum()
        fn_j = ((1 - preds[:, j]) * y_true[:, j]).sum()
        p_j  = tp_j / (tp_j + fp_j + 1e-8)
        r_j  = tp_j / (tp_j + fn_j + 1e-8)
        label_f1s.append(2 * p_j * r_j / (p_j + r_j + 1e-8))
    return float(micro_f1), float(np.mean(label_f1s))


def mean_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    n, ap_sum = y_true.shape[0], 0.0
    for i in range(n):
        ranked = np.argsort(scores[i])[::-1]
        rel    = y_true[i][ranked]
        if rel.sum() == 0:
            continue
        hits, prec_sum = 0, 0.0
        for r, r_val in enumerate(rel, 1):
            if r_val:
                hits     += 1
                prec_sum += hits / r
        ap_sum += prec_sum / rel.sum()
    return ap_sum / n


def mean_reciprocal_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    n, rr_sum = y_true.shape[0], 0.0
    for i in range(n):
        ranked = np.argsort(scores[i])[::-1]
        for rank, idx in enumerate(ranked, 1):
            if y_true[i][idx]:
                rr_sum += 1.0 / rank
                break
    return rr_sum / n


def macro_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    n_labels  = y_true.shape[1]
    auc_vals  = []
    for j in range(n_labels):
        pos = y_true[:, j].sum()
        neg = len(y_true) - pos
        if pos == 0 or neg == 0:
            continue
        order    = np.argsort(scores[:, j])[::-1]
        y_ord    = y_true[:, j][order]
        n_pos    = int(pos)
        rank_sum = np.where(y_ord == 1)[0].sum() + n_pos
        auc      = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * neg)
        auc_vals.append(auc)
    return float(np.mean(auc_vals)) if auc_vals else 0.0


def micro_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_flat = y_true.flatten()
    s_flat = scores.flatten()
    n_pos  = int(y_flat.sum())
    n_neg  = int(len(y_flat) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(s_flat)
    y_ord = y_flat[order]
    R     = (np.where(y_ord == 1)[0] + 1).sum()
    return float((R - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def label_ranking_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    n, lrap = y_true.shape[0], 0.0
    for i in range(n):
        true_idx = np.where(y_true[i] == 1)[0]
        if len(true_idx) == 0:
            continue
        ranked  = np.argsort(scores[i])[::-1]
        rank_of = {idx: r + 1 for r, idx in enumerate(ranked)}
        ap      = 0.0
        for l in true_idx:
            rank_l         = rank_of[l]
            above_and_true = sum(1 for l2 in true_idx if rank_of[l2] <= rank_l)
            ap += above_and_true / rank_l
        lrap += ap / len(true_idx)
    return lrap / n


def coverage_error(y_true: np.ndarray, scores: np.ndarray) -> float:
    n, total = y_true.shape[0], 0.0
    for i in range(n):
        true_idx = np.where(y_true[i] == 1)[0]
        if len(true_idx) == 0:
            continue
        ranked  = np.argsort(scores[i])[::-1]
        rank_of = {idx: r + 1 for r, idx in enumerate(ranked)}
        total  += max(rank_of[l] for l in true_idx)
    return total / n


def hits_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    n, hits = y_true.shape[0], 0
    for i in range(n):
        top_k = np.argsort(scores[i])[::-1][:k]
        if y_true[i][top_k].sum() > 0:
            hits += 1
    return hits / n


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    prefix: str = "",
) -> dict:
    px = f"{prefix}_" if prefix else ""
    m  = {}
    for k in [1, 3, 5]:
        m[f"{px}P@{k}"]    = precision_at_k(y_true, scores, k)
        m[f"{px}nDCG@{k}"] = ndcg_at_k(y_true, scores, k)
    micro_f1, macro_f1 = f1_scores(y_true, scores, threshold)
    m[f"{px}micro_F1"]           = micro_f1
    m[f"{px}macro_F1"]           = macro_f1
    m[f"{px}MRR"]                = mean_reciprocal_rank(y_true, scores)
    m[f"{px}mAP"]                = mean_average_precision(y_true, scores)
    m[f"{px}LRAP"]               = label_ranking_average_precision(y_true, scores)
    m[f"{px}micro_AUROC"]        = micro_auroc(y_true, scores)
    m[f"{px}macro_AUROC"]        = macro_auroc(y_true, scores)
    m[f"{px}coverage_error"]     = coverage_error(y_true, scores)
    m[f"{px}Hits@5"]             = hits_at_k(y_true, scores, 5)
    m[f"{px}Hits@10"]            = hits_at_k(y_true, scores, 10)
    m[f"{px}avg_labels_per_doc"] = float(y_true.sum(axis=1).mean())
    m[f"{px}total_docs"]         = int(y_true.shape[0])
    m[f"{px}total_labels_in_vocab"] = int(y_true.shape[1])
    return m


def log_metrics(metrics: dict, title: str):
    logger.info(f"\n{'='*58}")
    logger.info(f"  {title}")
    logger.info(f"{'='*58}")
    for k, v in metrics.items():
        logger.info(
            f"  {k:<38} {v:.4f}" if isinstance(v, float)
            else f"  {k:<38} {v}"
        )


# ===========================================================================
# 9. GZSL split helper
# ===========================================================================

def split_by_seen_unseen(
    y_true: np.ndarray,
    scores: np.ndarray,
    label_map: Dict[str, int],
    seen_codes: Set[str],
    unseen_codes: Set[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seen_idx   = sorted([i for lbl, i in label_map.items() if lbl in seen_codes])
    unseen_idx = sorted([i for lbl, i in label_map.items() if lbl in unseen_codes])

    def _slice(mat, idx):
        return mat[:, idx] if idx else np.zeros((mat.shape[0], 0), dtype=mat.dtype)

    return (
        _slice(y_true, seen_idx),   _slice(scores, seen_idx),
        _slice(y_true, unseen_idx), _slice(scores, unseen_idx),
    )


# ===========================================================================
# 10. Train / eval epoch
# ===========================================================================

def run_epoch(
    model: PubMedBERTWithTLV9,
    loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    scaler,
    grad_clip: float,
    grad_accum: int,
    training: bool,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.train(training)
    total_loss = 0.0
    all_y_true, all_scores = [], []

    if training:
        optimizer.zero_grad()

    for step, batch in enumerate(
        tqdm(loader, desc="train" if training else "eval", leave=False)
    ):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        ctx = (
            torch.cuda.amp.autocast() if (scaler and training)
            else torch.no_grad() if not training
            else torch.enable_grad()
        )
        with ctx:
            logits = model(ids, mask)
            loss   = loss_fn(logits, lbls)

        if training:
            scaled = loss / grad_accum
            if scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (step + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                trainable = [p for p in model.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable, grad_clip)
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item()
        with torch.no_grad():
            all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_y_true.append(lbls.cpu().numpy())

    return (
        total_loss / max(len(loader), 1),
        np.concatenate(all_y_true,  axis=0),
        np.concatenate(all_scores, axis=0),
    )


# ===========================================================================
# 11. Checkpoint save / load
# ===========================================================================

def save_checkpoint(
    model: PubMedBERTWithTLV9,
    tokenizer,
    seen_map: Dict[str, int],
    unseen_map: Dict[str, int],
    config: dict,
    output_dir: Path,
):
    model_dir = output_dir / "best_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(str(model_dir))
    torch.save(model.classifier.state_dict(), model_dir / "hybrid_classifier.pt")
    # Save translation layer: directions + learned log_gamma
    torch.save(
        {
            "V":         model.translation.V.cpu(),
            "log_gamma": model.translation.log_gamma.detach().cpu(),
        },
        model_dir / "translation_layer.pt",
    )
    with open(model_dir / "seen_map.json",   "w") as f:
        json.dump(seen_map,   f)
    with open(model_dir / "unseen_map.json", "w") as f:
        json.dump(unseen_map, f)
    with open(model_dir / "config.json",     "w") as f:
        json.dump(config, f, indent=2)

    gamma_val = float(F.softplus(model.translation.log_gamma).item())
    logger.info(
        f"Checkpoint saved → {model_dir}  "
        f"(gamma={gamma_val:.4f}, log_gamma={model.translation.log_gamma.item():.4f})"
    )


def load_checkpoint(
    output_dir: Path,
    device: torch.device,
) -> Tuple["PubMedBERTWithTLV9", object, Dict[str, int], Dict[str, int], dict]:
    model_dir = output_dir / "best_model"

    with open(model_dir / "config.json") as f:
        config = json.load(f)
    with open(model_dir / "seen_map.json") as f:
        seen_map: Dict[str, int] = json.load(f)
    with open(model_dir / "unseen_map.json") as f:
        unseen_map: Dict[str, int] = json.load(f)

    tokenizer  = AutoTokenizer.from_pretrained(str(model_dir))
    bert_model = AutoModel.from_pretrained(PUBMEDBERT_CKPT)

    gamma_init = config.get("gamma_init", -2.3)
    model      = PubMedBERTWithTLV9(
        bert_model,
        num_seen=len(seen_map),
        num_unseen=len(unseen_map),
        gamma_init=gamma_init,
    ).to(device)

    model.classifier.load_state_dict(
        torch.load(model_dir / "hybrid_classifier.pt", map_location=device)
    )

    tl_path = model_dir / "translation_layer.pt"
    if tl_path.exists():
        tl_state = torch.load(tl_path, map_location=device)
        model.translation.V.copy_(tl_state["V"].to(device))
        model.translation.log_gamma.data.copy_(tl_state["log_gamma"].to(device))
        gamma_val = float(F.softplus(model.translation.log_gamma).item())
        logger.info(f"Translation layer loaded (gamma={gamma_val:.4f})")
    else:
        logger.warning("translation_layer.pt not found — translation is a no-op.")

    model.top_k         = config.get("top_k", 0)
    model.warmup_active = False   # always off at test time

    return model, tokenizer, seen_map, unseen_map, config


# ===========================================================================
# 12. Training orchestration
# ===========================================================================

def run_train(args, device: torch.device):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info(f"DOMAIN: {args.domain}  |  MODE: train  |  v9")
    logger.info("=" * 65)
    logger.info(
        f"Warm-up epochs : {args.warmup_epochs}  "
        f"(translation OFF for first {args.warmup_epochs} epoch(s))"
    )
    logger.info(
        f"Total epochs   : {args.epochs}  "
        f"(translation ON for epochs {args.warmup_epochs + 1}–{args.epochs})"
    )
    logger.info(f"Top-k gate     : {args.top_k}  (0 = all gates active)")
    logger.info(f"Gamma init     : softplus({args.gamma_init:.3f}) = "
                f"{float(F.softplus(torch.tensor(args.gamma_init))):.4f}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_items = load_json_items(args.train_path)
    val_items   = load_json_items(args.val_path)
    logger.info(f"Train: {len(train_items)} docs  |  Val: {len(val_items)} docs")

    with open(args.unseen_codes_path) as f:
        unseen_codes: List[str] = json.load(f)

    seen_map, unseen_map = build_label_maps(train_items, args.label_mode, unseen_codes)
    num_seen   = len(seen_map)
    num_unseen = len(unseen_map)
    full_map   = dict(seen_map)
    for lbl, idx in unseen_map.items():
        full_map[lbl] = idx + num_seen

    tokenizer  = AutoTokenizer.from_pretrained(PUBMEDBERT_CKPT)
    train_ds   = MeshDataset(train_items, tokenizer, seen_map,
                             args.label_mode, args.max_len, "train")
    val_ds     = MeshDataset(val_items,   tokenizer, seen_map,
                             args.label_mode, args.max_len, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ── Build model ───────────────────────────────────────────────────────
    bert_model = AutoModel.from_pretrained(PUBMEDBERT_CKPT)
    model      = PubMedBERTWithTLV9(
        bert_model, num_seen, num_unseen, gamma_init=args.gamma_init
    ).to(device)
    model.top_k = args.top_k

    # ── Step 1: Unseen label embeddings ───────────────────────────────────
    logger.info("Step 1: Building label embeddings (unseen) ...")
    E_all    = build_label_embeddings(
        full_map, args.nodes_csv_path, model.bert, tokenizer, device
    )
    E_unseen = E_all[num_seen:, :]
    model.classifier.set_unseen_embeddings(E_unseen)
    del E_all, E_unseen

    # ── Step 2: Spurious direction estimation ─────────────────────────────
    logger.info("Step 2: Estimating spurious directions ...")
    abstract_mesh_df   = load_abstract_mesh(args.abstract_mesh_path)
    embeddings, pmids  = collect_cls_embeddings(
        model.bert, tokenizer, train_items, seen_map,
        args.label_mode, args.max_len, args.batch_size, device,
    )
    # Estimate directions only for seen labels
    directions_seen = estimate_spurious_directions(
        embeddings, pmids, seen_map, abstract_mesh_df
    )
    model.translation.set_directions(directions_seen)
    del embeddings

    # ── Step 3: Optimizer + scheduler ─────────────────────────────────────
    trainable   = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info(f"Trainable params: {n_trainable:,}  "
                f"(seen_head + query_adapter + log_tau + log_gamma)")

    if args.use_pos_weight:
        pw      = compute_pos_weight(train_items, seen_map, args.label_mode).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        logger.info("Using pos_weight in BCEWithLogitsLoss.")
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer   = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler   = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler      = (
        torch.cuda.amp.GradScaler()
        if args.fp16 and device.type == "cuda" else None
    )

    config = {
        "domain":            args.domain,
        "label_mode":        args.label_mode,
        "max_len":           args.max_len,
        "batch_size":        args.batch_size,
        "lr":                args.lr,
        "epochs":            args.epochs,
        "warmup_epochs":     args.warmup_epochs,
        "warmup_ratio":      args.warmup_ratio,
        "top_k":             args.top_k,
        "gamma_init":        args.gamma_init,
        "seed":              args.seed,
        "use_pos_weight":    args.use_pos_weight,
        "fp16":              args.fp16,
        "grad_accum_steps":  args.grad_accum_steps,
        "num_seen":          num_seen,
        "num_unseen":        num_unseen,
        "bert_ckpt":         PUBMEDBERT_CKPT,
        "model_type":        "pubmedbert_hybrid_tl_v9",
    }

    log_path       = output_dir / "train_log.jsonl"
    best_ndcg5     = -1.0
    best_epoch     = -1
    patience_count = 0

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Toggle warm-up phase
        was_warmup = model.warmup_active
        model.warmup_active = (epoch <= args.warmup_epochs)
        if was_warmup and not model.warmup_active:
            gamma_now = float(F.softplus(model.translation.log_gamma).item())
            logger.info(
                f">>> Warm-up complete. Translation ACTIVE from epoch {epoch}. "
                f"gamma={gamma_now:.4f}"
            )

        model.train_mode = True
        train_loss, _, _ = run_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, scaler, args.grad_clip, args.grad_accum_steps,
            training=True,
        )

        model.train_mode = True   # val also uses seen-only logits for speed
        val_loss, val_y, val_s = run_epoch(
            model, val_loader, None, None, loss_fn,
            device, None, args.grad_clip, 1,
            training=False,
        )

        val_m   = compute_metrics(val_y, val_s, args.threshold, prefix="val")
        elapsed = time.time() - t0
        gamma_val = float(F.softplus(model.translation.log_gamma).item())

        phase_tag = "WARMUP" if model.warmup_active else "ACTIVE"
        logger.info(
            f"Epoch {epoch:02d}/{args.epochs} [{phase_tag}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"nDCG@5={val_m['val_nDCG@5']:.4f}  P@1={val_m['val_P@1']:.4f}  "
            f"F1={val_m['val_micro_F1']:.4f}  "
            f"gamma={gamma_val:.4f}  ({elapsed:.1f}s)"
        )

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "phase": phase_tag,
                "gamma": gamma_val,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                **val_m,
            }) + "\n")

        if val_m["val_nDCG@5"] > best_ndcg5:
            best_ndcg5     = val_m["val_nDCG@5"]
            best_epoch     = epoch
            patience_count = 0
            save_checkpoint(model, tokenizer, seen_map, unseen_map, config, output_dir)
            logger.info(f"  ✓ New best val nDCG@5={best_ndcg5:.4f} at epoch {best_epoch}")
        else:
            patience_count += 1
            logger.info(f"  No improvement ({patience_count}/{args.patience})")
            if patience_count >= args.patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    logger.info(
        f"\nTraining done [{args.domain}]. "
        f"Best val nDCG@5={best_ndcg5:.4f} at epoch {best_epoch}."
        f"\nCheckpoint: {output_dir}/best_model"
    )


# ===========================================================================
# 13. Test / inference
# ===========================================================================

@torch.inference_mode()
def infer(
    model: PubMedBERTWithTLV9,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    all_y, all_s = [], []
    for batch in tqdm(loader, desc="inference", leave=False):
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        all_s.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(batch["labels"].numpy())
    return np.concatenate(all_y, axis=0), np.concatenate(all_s, axis=0)


def run_test(args, device: torch.device):
    output_dir = Path(args.output_dir)

    logger.info("=" * 65)
    logger.info(f"DOMAIN: {args.domain}  |  MODE: test  |  EXPERIMENT: {args.experiment}")
    logger.info("=" * 65)

    model, tokenizer, seen_map, unseen_map, config = load_checkpoint(output_dir, device)
    label_mode = config.get("label_mode", args.label_mode)
    max_len    = config.get("max_len",    args.max_len)

    num_seen = len(seen_map)
    full_map = dict(seen_map)
    for lbl, idx in unseen_map.items():
        full_map[lbl] = idx + num_seen

    # Test always uses full pipeline: translation active, forward_all
    model.warmup_active = False
    model.train_mode    = False

    test_items = load_json_items(args.test_path)
    logger.info(f"Test docs: {len(test_items)}")

    test_ds = MeshDataset(
        test_items, tokenizer, full_map,
        label_mode, max_len,
        split_name=f"test_{args.experiment}",
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    y_true, scores = infer(model, test_loader, device)
    logger.info(
        f"Score matrix: {scores.shape}  "
        f"(seen={num_seen}, unseen={len(unseen_map)})"
    )

    # Log learned gamma at test time
    gamma_val = float(F.softplus(model.translation.log_gamma).item())
    logger.info(f"Learned gamma at test time: {gamma_val:.4f}")

    all_results = {}

    if args.experiment == "supervised":
        s_seen = scores[:, :num_seen]
        y_seen = y_true[:, :num_seen]
        metrics = compute_metrics(y_seen, s_seen, args.threshold, prefix="")
        log_metrics(metrics, "SUPERVISED — Test Results")
        all_results["supervised"] = metrics

    elif args.experiment == "zeroshot":
        in_vocab, out_vocab = 0, 0
        for item in test_items:
            for lbl in extract_labels_from_item(item, label_mode):
                if lbl in full_map:
                    in_vocab += 1
                else:
                    out_vocab += 1
        logger.info(
            f"Zero-shot label coverage:"
            f"\n  In vocab  : {in_vocab}"
            f"\n  Not in vocab: {out_vocab}"
        )
        metrics = compute_metrics(y_true, scores, args.threshold, prefix="")
        metrics["labels_in_vocab"]     = in_vocab
        metrics["labels_not_in_vocab"] = out_vocab
        log_metrics(metrics, "ZERO-SHOT — Test Results")
        all_results["zeroshot"] = metrics

    elif args.experiment == "gzsl":
        if not args.seen_codes_path or not args.unseen_codes_path:
            raise ValueError("--experiment gzsl requires --seen_codes_path and --unseen_codes_path")

        with open(args.seen_codes_path)   as f: seen_codes: Set[str]   = set(json.load(f))
        with open(args.unseen_codes_path) as f: unseen_codes_set: Set[str] = set(json.load(f))

        m_overall = compute_metrics(y_true, scores, args.threshold, prefix="overall")
        y_seen_m, s_seen_m, y_unseen_m, s_unseen_m = split_by_seen_unseen(
            y_true, scores, full_map, seen_codes, unseen_codes_set
        )
        m_seen   = compute_metrics(y_seen_m,   s_seen_m,   args.threshold, prefix="seen")   if y_seen_m.shape[1]   > 0 else {}
        m_unseen = compute_metrics(y_unseen_m, s_unseen_m, args.threshold, prefix="unseen") if y_unseen_m.shape[1] > 0 else {}

        log_metrics(m_overall, "GZSL — Overall")
        if m_seen:
            log_metrics(m_seen, "GZSL — Seen Codes Only")
        if m_unseen:
            log_metrics(m_unseen, "GZSL — Unseen Codes Only")

        all_results.update({"gzsl_overall": m_overall, "gzsl_seen": m_seen, "gzsl_unseen": m_unseen})

    elif args.experiment == "tail":
        # ------------------------------------------------------------------
        # Tail-label evaluation
        # ------------------------------------------------------------------
        # Ground truth is masked to rare labels only (head labels zeroed out
        # in y_true). Scores are left intact — the model still ranks over the
        # full seen label space. This correctly penalises the model for
        # ranking head labels above rare ones when the document has rare
        # ground-truth labels.
        # ------------------------------------------------------------------
        if not args.tail_label_ids_path:
            raise ValueError("--experiment tail requires --tail_label_ids_path")

        with open(args.tail_label_ids_path) as f:
            tail_ids: Set[str] = set(json.load(f))

        # Work in seen-label space only (mirrors supervised evaluation)
        s_seen = scores[:, :num_seen]
        y_seen = y_true[:, :num_seen]

        # Build boolean mask: True = rare label column
        seen_idx_to_lbl = {v: k for k, v in seen_map.items()}
        tail_col_mask = np.array(
            [seen_idx_to_lbl.get(i, "") in tail_ids for i in range(num_seen)],
            dtype=bool,
        )
        n_tail_cols = tail_col_mask.sum()
        n_head_cols = num_seen - n_tail_cols
        logger.info(
            f"Tail-label mask: {n_tail_cols} rare cols, "
            f"{n_head_cols} head cols zeroed in y_true  "
            f"(seen space = {num_seen})"
        )

        if n_tail_cols == 0:
            raise RuntimeError(
                "No tail labels found in seen label space. "
                "Check that tail_label_ids.json matches --label_mode unique_id."
            )

        # Zero out head-label ground truth; scores unchanged
        y_tail = y_seen.copy()
        y_tail[:, ~tail_col_mask] = 0.0

        # Filter to docs that have at least one rare ground-truth label
        has_tail = y_tail.sum(axis=1) > 0
        n_tail_docs = has_tail.sum()
        logger.info(
            f"Docs with ≥1 rare ground-truth label: "
            f"{n_tail_docs} / {len(y_tail)}"
        )

        if n_tail_docs == 0:
            raise RuntimeError(
                "No documents have rare ground-truth labels in this test file. "
                "Verify that --test_path points to test_tail.json."
            )

        y_eval = y_tail[has_tail]
        s_eval = s_seen[has_tail]

        metrics = compute_metrics(y_eval, s_eval, args.threshold, prefix="")
        metrics["tail_label_cols"]     = int(n_tail_cols)
        metrics["head_label_cols_zeroed"] = int(n_head_cols)
        metrics["tail_docs_evaluated"] = int(n_tail_docs)
        metrics["total_docs_in_file"]  = int(len(y_tail))
        log_metrics(metrics, "TAIL-LABEL — Test Results (head labels masked from y_true)")
        all_results["tail"] = metrics

    results_path = output_dir / f"test_results_{args.experiment}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved → {results_path}")


# ===========================================================================
# 14. Args + main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PubMedBERT + Spurious Translation Layer v9"
    )

    # Core
    p.add_argument("--mode",       required=True, choices=["train", "test"])
    p.add_argument("--experiment", required=True, choices=["supervised", "zeroshot", "gzsl", "tail"])
    p.add_argument("--domain",     required=True, choices=["neurology", "immunology", "embryology"])
    p.add_argument("--output_dir", required=True)

    # Paths
    p.add_argument("--train_path",           default=None)
    p.add_argument("--val_path",             default=None)
    p.add_argument("--test_path",            default=None)
    p.add_argument("--abstract_mesh_path",   default=None)
    p.add_argument("--nodes_csv_path",       default=None)
    p.add_argument("--unseen_codes_path",    default=None)
    p.add_argument("--seen_codes_path",      default=None)
    p.add_argument(
        "--tail_label_ids_path", default=None,
        help="Path to tail_label_ids.json produced by create_tail_label_set.py. "
             "Required for --experiment tail.",
    )
    p.add_argument(
        "--tail_mask_head_labels", action="store_true",
        help="When set, zero out head-label columns in y_true before computing "
             "tail metrics (scores are always over the full seen label space). "
             "This is the correct evaluation: the model ranks all labels, but "
             "precision/recall is measured only against rare ground-truth labels.",
    )

    # Label
    p.add_argument("--label_mode", default="unique_id",
                   choices=["unique_id", "tree_numbers", "heading", "combined"])
    p.add_argument("--max_len",    type=int, default=256)

    # Training HPs (all tunable per domain)
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--lr",               type=float, default=1e-3)
    p.add_argument("--epochs",           type=int,   default=20)
    p.add_argument("--patience",         type=int,   default=5)
    p.add_argument("--grad_accum_steps", type=int,   default=1)
    p.add_argument("--grad_clip",        type=float, default=1.0)
    p.add_argument("--warmup_ratio",     type=float, default=0.1,
                   help="LR scheduler warm-up as fraction of total steps.")
    p.add_argument("--use_pos_weight",   action="store_true")
    p.add_argument("--fp16",             action="store_true")
    p.add_argument("--threshold",        type=float, default=0.5)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)

    # v9-specific HPs
    p.add_argument(
        "--warmup_epochs", type=int, default=5,
        help="Number of epochs to train WITHOUT translation (gate disabled). "
             "Allows the classifier to calibrate before the gate activates. "
             "Recommended: 3–8 depending on domain size.",
    )
    p.add_argument(
        "--top_k", type=int, default=0,
        help="Top-k gate sparsification: keep only the k highest sigmoid scores "
             "per document; zero out the rest. 0 = disabled (all gates active). "
             "Suggested starting point: avg labels per doc for the domain "
             "(neurology~9, immunology~12, embryology~12).",
    )
    p.add_argument(
        "--gamma_init", type=float, default=-2.3,
        help="Initial value of log_gamma (log-space). "
             "softplus(gamma_init) = initial gamma. "
             "Default -2.3 → gamma≈0.1 (conservative). "
             "Lower = more conservative, higher = more aggressive debiasing at start.",
    )

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Domain: {args.domain}  |  v9")

    if args.mode == "train":
        for flag, name in [
            (args.train_path,         "--train_path"),
            (args.val_path,           "--val_path"),
            (args.abstract_mesh_path, "--abstract_mesh_path"),
            (args.nodes_csv_path,     "--nodes_csv_path"),
            (args.unseen_codes_path,  "--unseen_codes_path"),
        ]:
            if not flag:
                raise ValueError(f"--mode train requires {name}")
        run_train(args, device)

    elif args.mode == "test":
        if not args.test_path:
            raise ValueError("--mode test requires --test_path")
        run_test(args, device)


if __name__ == "__main__":
    main()
