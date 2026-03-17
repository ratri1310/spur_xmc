"""
test_pubmedbert_mesh_tl_v9.py
==============================
Standalone test/inference script for the PubMedBERT + Spurious Translation Layer (v9).
Loads a saved checkpoint from train_pubmedbert_mesh_tl_v9.py and runs evaluation.

Supported experiment modes:
    supervised  — evaluate on seen labels
    tail        — evaluate on rare labels only (head labels masked from y_true)
    zeroshot    — evaluate on unseen labels only
    gzsl        — evaluate on seen + unseen labels


"""

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

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

def _extract_entry_fields(entry: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
            return items
        list_values = [(k, v) for k, v in raw.items() if isinstance(v, list)]
        if len(list_values) == 1:
            return [e for e in list_values[0][1] if isinstance(e, dict)]
        raise ValueError(f"Ambiguous JSON structure.")
    raise ValueError(f"Unexpected JSON root type: {type(raw).__name__}")


# ===========================================================================
# 3. Dataset
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
                f"[{split_name}] {dropped} label occurrences not in label_map (dropped)."
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
# 4. Model
# ===========================================================================

def top_k_gate(s: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return s
    batch, L  = s.shape
    k_clamped = min(k, L)
    topk_vals, topk_idx = torch.topk(s, k_clamped, dim=1)
    s_sparse = torch.zeros_like(s)
    s_sparse.scatter_(1, topk_idx, topk_vals)
    return s_sparse


class SpuriousTranslationLayerV9(nn.Module):
    def __init__(self, num_seen: int, hidden_size: int = 768, gamma_init: float = -2.3):
        super().__init__()
        self.register_buffer("V", torch.zeros(num_seen, hidden_size))
        self.log_gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.num_seen  = num_seen

    def forward(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        top_k: int = 0,
        active: bool = True,
    ) -> torch.Tensor:
        if not active:
            return h
        gamma      = F.softplus(self.log_gamma)
        s_sparse   = top_k_gate(s, top_k)
        proj       = h @ self.V.T
        raw_signal = (s_sparse * proj) @ self.V
        return h - gamma * raw_signal


class HybridClassifier(nn.Module):
    def __init__(self, num_seen: int, num_unseen: int, hidden_size: int = 768):
        super().__init__()
        self.num_seen   = num_seen
        self.num_unseen = num_unseen
        self.seen_head  = nn.Linear(hidden_size, num_seen)
        self.query_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.LayerNorm(hidden_size),
        )
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
        self.register_buffer("E_unseen", torch.zeros(num_unseen, hidden_size))

    def forward_seen(self, z: torch.Tensor) -> torch.Tensor:
        return self.seen_head(z)

    def forward_all(self, z: torch.Tensor) -> torch.Tensor:
        seen_logits   = self.seen_head(z)
        q             = F.normalize(self.query_adapter(z), dim=1)
        tau           = self.log_tau.exp().clamp(min=1e-4)
        unseen_logits = (q @ self.E_unseen.T) / tau
        return torch.cat([seen_logits, unseen_logits], dim=1)


class PubMedBERTWithTLV9(nn.Module):
    def __init__(self, bert_model, num_seen: int, num_unseen: int, gamma_init: float = -2.3):
        super().__init__()
        hidden           = bert_model.config.hidden_size
        self.bert        = bert_model
        self.translation = SpuriousTranslationLayerV9(num_seen, hidden, gamma_init)
        self.classifier  = HybridClassifier(num_seen, num_unseen, hidden)
        self.num_seen    = num_seen
        self.num_unseen  = num_unseen
        self.train_mode  = False
        self.warmup_active = False
        self.top_k         = 0
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            h   = out.last_hidden_state[:, 0, :].float().detach()
        with torch.no_grad():
            W_gate = self.classifier.seen_head.weight.detach()
            b_gate = self.classifier.seen_head.bias.detach()
            s      = torch.sigmoid(h @ W_gate.T + b_gate)
        z = self.translation(h, s, top_k=self.top_k, active=not self.warmup_active)
        if self.train_mode:
            return self.classifier.forward_seen(z)
        else:
            return self.classifier.forward_all(z)


# ===========================================================================
# 5. Checkpoint loading
# ===========================================================================

def load_checkpoint(
    output_dir: Path,
    device: torch.device,
) -> Tuple[PubMedBERTWithTLV9, object, Dict[str, int], Dict[str, int], dict]:
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

    model.top_k        = config.get("top_k", 0)
    model.warmup_active = False
    model.train_mode    = False

    return model, tokenizer, seen_map, unseen_map, config


# ===========================================================================
# 6. Metrics
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
        ideal = sum(1.0 / math.log2(j + 2) for j in range(min(int(y_true[i].sum()), k)))
        ndcg += dcg / ideal if ideal > 0 else 0.0
    return ndcg / n


def f1_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[float, float]:
    preds = (scores >= threshold).astype(np.float32)
    tp = (preds * y_true).sum()
    fp = (preds * (1 - y_true)).sum()
    fn = ((1 - preds) * y_true).sum()
    p  = tp / (tp + fp + 1e-8)
    r  = tp / (tp + fn + 1e-8)
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


def macro_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    n_labels = y_true.shape[1]
    auc_vals = []
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
    m[f"{px}micro_F1"]              = micro_f1
    m[f"{px}macro_F1"]              = macro_f1
    m[f"{px}MRR"]                   = mean_reciprocal_rank(y_true, scores)
    m[f"{px}mAP"]                   = mean_average_precision(y_true, scores)
    m[f"{px}micro_AUROC"]           = micro_auroc(y_true, scores)
    m[f"{px}macro_AUROC"]           = macro_auroc(y_true, scores)
    m[f"{px}coverage_error"]        = coverage_error(y_true, scores)
    m[f"{px}Hits@5"]                = hits_at_k(y_true, scores, 5)
    m[f"{px}Hits@10"]               = hits_at_k(y_true, scores, 10)
    m[f"{px}avg_labels_per_doc"]    = float(y_true.sum(axis=1).mean())
    m[f"{px}total_docs"]            = int(y_true.shape[0])
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
# 7. GZSL split helper
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
# 8. Inference
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


# ===========================================================================
# 9. Main test runner
# ===========================================================================

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

    model.warmup_active = False
    model.train_mode    = False

    # Log learned gamma
    gamma_val = float(F.softplus(model.translation.log_gamma).item())
    logger.info(f"Learned gamma : {gamma_val:.4f}")
    logger.info(f"Top-k         : {model.top_k}")

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
    logger.info(f"Score matrix: {scores.shape}  (seen={num_seen}, unseen={len(unseen_map)})")

    all_results = {
        "gamma": gamma_val,
        "top_k": model.top_k,
    }

    # ── Supervised ──────────────────────────────────────────────────────────
    if args.experiment == "supervised":
        s_seen = scores[:, :num_seen]
        y_seen = y_true[:, :num_seen]
        metrics = compute_metrics(y_seen, s_seen, args.threshold, prefix="")
        log_metrics(metrics, "SUPERVISED — Test Results")
        all_results["supervised"] = metrics

    # ── Zero-shot ────────────────────────────────────────────────────────────
    elif args.experiment == "zeroshot":
        in_vocab, out_vocab = 0, 0
        for item in test_items:
            for lbl in extract_labels_from_item(item, label_mode):
                if lbl in full_map:
                    in_vocab += 1
                else:
                    out_vocab += 1
        logger.info(f"Zero-shot label coverage: in_vocab={in_vocab}, not_in_vocab={out_vocab}")
        metrics = compute_metrics(y_true, scores, args.threshold, prefix="")
        metrics["labels_in_vocab"]     = in_vocab
        metrics["labels_not_in_vocab"] = out_vocab
        log_metrics(metrics, "ZERO-SHOT — Test Results")
        all_results["zeroshot"] = metrics

    # ── GZSL ─────────────────────────────────────────────────────────────────
    elif args.experiment == "gzsl":
        if not args.seen_codes_path or not args.unseen_codes_path:
            raise ValueError("--experiment gzsl requires --seen_codes_path and --unseen_codes_path")
        with open(args.seen_codes_path)   as f: seen_codes: Set[str]      = set(json.load(f))
        with open(args.unseen_codes_path) as f: unseen_codes_set: Set[str] = set(json.load(f))

        m_overall = compute_metrics(y_true, scores, args.threshold, prefix="overall")
        y_seen_m, s_seen_m, y_unseen_m, s_unseen_m = split_by_seen_unseen(
            y_true, scores, full_map, seen_codes, unseen_codes_set
        )
        m_seen   = compute_metrics(y_seen_m,   s_seen_m,   args.threshold, prefix="seen")   if y_seen_m.shape[1]   > 0 else {}
        m_unseen = compute_metrics(y_unseen_m, s_unseen_m, args.threshold, prefix="unseen") if y_unseen_m.shape[1] > 0 else {}

        log_metrics(m_overall, "GZSL — Overall")
        if m_seen:   log_metrics(m_seen,   "GZSL — Seen Only")
        if m_unseen: log_metrics(m_unseen, "GZSL — Unseen Only")
        all_results.update({"gzsl_overall": m_overall, "gzsl_seen": m_seen, "gzsl_unseen": m_unseen})

    # ── Tail-label ───────────────────────────────────────────────────────────
    elif args.experiment == "tail":
        if not args.tail_label_ids_path:
            raise ValueError("--experiment tail requires --tail_label_ids_path")

        with open(args.tail_label_ids_path) as f:
            tail_ids: Set[str] = set(json.load(f))

        s_seen = scores[:, :num_seen]
        y_seen = y_true[:, :num_seen]

        seen_idx_to_lbl = {v: k for k, v in seen_map.items()}
        tail_col_mask   = np.array(
            [seen_idx_to_lbl.get(i, "") in tail_ids for i in range(num_seen)],
            dtype=bool,
        )
        n_tail_cols = tail_col_mask.sum()
        n_head_cols = num_seen - n_tail_cols
        logger.info(f"Tail cols: {n_tail_cols}  Head cols zeroed: {n_head_cols}")

        if n_tail_cols == 0:
            raise RuntimeError("No tail labels found in seen label space.")

        y_tail = y_seen.copy()
        y_tail[:, ~tail_col_mask] = 0.0

        has_tail    = y_tail.sum(axis=1) > 0
        n_tail_docs = has_tail.sum()
        logger.info(f"Docs with ≥1 rare ground-truth label: {n_tail_docs} / {len(y_tail)}")

        if n_tail_docs == 0:
            raise RuntimeError("No documents have rare ground-truth labels in this test file.")

        y_eval  = y_tail[has_tail]
        s_eval  = s_seen[has_tail]
        metrics = compute_metrics(y_eval, s_eval, args.threshold, prefix="")
        metrics["tail_label_cols"]        = int(n_tail_cols)
        metrics["head_label_cols_zeroed"] = int(n_head_cols)
        metrics["tail_docs_evaluated"]    = int(n_tail_docs)
        metrics["total_docs_in_file"]     = int(len(y_tail))
        log_metrics(metrics, "TAIL-LABEL — Test Results (head labels masked)")
        all_results["tail"] = metrics

    results_path = output_dir / f"test_results_{args.experiment}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved → {results_path}")


# ===========================================================================
# 10. Args + main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Test PubMedBERT + Spurious Translation Layer v9")

    p.add_argument("--experiment", required=True, choices=["supervised", "zeroshot", "gzsl", "tail"])
    p.add_argument("--domain",     required=True, choices=["neurology", "immunology", "embryology"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--test_path",  required=True)

    p.add_argument("--seen_codes_path",      default=None)
    p.add_argument("--unseen_codes_path",    default=None)
    p.add_argument("--tail_label_ids_path",  default=None)
    p.add_argument("--tail_mask_head_labels", action="store_true")

    p.add_argument("--label_mode",  default="unique_id",
                   choices=["unique_id", "tree_numbers", "heading", "combined"])
    p.add_argument("--max_len",     type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--threshold",   type=float, default=0.5)
    p.add_argument("--num_workers", type=int,   default=4)

    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}  |  Domain: {args.domain}")
    run_test(args, device)


if __name__ == "__main__":
    main()
