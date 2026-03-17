# SPUR-XMC: Context-Adaptive Spurious Correlation Mitigation for Extreme Multi-Label Classification of Scientific Literature
# PubMedBERT + Context-Adaptive Spurious Correlation Translation Layer


This repository contains the implementation of the Spurious Translation Layer for MeSH multi-label classification of PubMed abstracts. The model uses a frozen PubMedBERT encoder with a learned gated translation layer that identifies and corrects spurious directions in the representation space.

---

## Architecture

```
BERT (frozen)
    → CLS embedding h [batch, 768]
    → SpuriousTranslationLayer (v9)
        Phase 1 (warm-up): z = h          (translation OFF)
        Phase 2 (remaining epochs):
            s    = sigmoid(W·h + b)        [batch, num_seen]
            s̃   = top_k_gate(s, k)         [batch, num_seen]
            z    = h - softplus(γ) * (s̃ ⊙ (h @ V.T)) @ V
    → HybridClassifier
        seen:   Linear(768 → num_seen)     (trainable)

```

---

## Requirements

```bash
pip install -r requirements.txt
```

Or use the provided `requirements.txt` directly.

---

## Input Data Format

### Training / Validation / Test JSON
Each file is a list of records:
```json
[
  {
    "pmid": "12345678",
    "title": "...",
    "abstractText": "...",
    "meshMajorEnhanced": [
      {"mesh_heading": "...", "tree_numbers": "...", "unique_id": "D001234"}
    ]
  }
]
```

### Abstract-MeSH CSV (`abstract_mesh_path`)
Required columns: `pmid`, `mesh_id`, `is_spurious`
```
pmid,mesh_id,is_spurious
12345678,D001234,True
12345678,D005678,False
```
This file provides the KG-derived spuriousness indicator $c_{i\ell}$ used to estimate the spurious direction matrix $V$.

### Nodes CSV (`nodes_csv_path`)
Required columns: `mesh_id`, `pref_name`
```
mesh_id,pref_name
D001234,Brain Diseases
```


## Training

### Neurology
```bash
TOKENIZERS_PARALLELISM=false python train_pubmedbert_mesh_tl_v9.py \
    --mode train --experiment supervised --domain neurology \
    --train_path         /path/to/splits/neurology/train.json \
    --val_path           /path/to/splits/neurology/val.json \
    --abstract_mesh_path /path/to/kg/neurology/splits/train_abstract_mesh.csv \
    --nodes_csv_path     /path/to/kg/neurology/nodes.csv \
    --unseen_codes_path  /path/to/splits/neurology/unseen_codes.json \
    --output_dir         /path/to/runs/neurology_tl_v9 \
    --label_mode unique_id --epochs 20 --warmup_epochs 5 \
    --batch_size 32 --lr 1e-3 --top_k 0 --fp16
```

### Immunology
```bash
TOKENIZERS_PARALLELISM=false python train_pubmedbert_mesh_tl_v9.py \
    --mode train --experiment supervised --domain immunology \
    --train_path         /path/to/splits/immunology/train.json \
    --val_path           /path/to/splits/immunology/val.json \
    --abstract_mesh_path /path/to/kg/immunology/splits/train_abstract_mesh.csv \
    --nodes_csv_path     /path/to/kg/immunology/nodes.csv \
    --unseen_codes_path  /path/to/splits/immunology/unseen_codes.json \
    --output_dir         /path/to/runs/immunology_tl_v9 \
    --label_mode unique_id --epochs 20 --warmup_epochs 5 \
    --batch_size 32 --lr 1e-3 --top_k 0 --fp16
```

### Embryology
```bash
TOKENIZERS_PARALLELISM=false python train_pubmedbert_mesh_tl_v9.py \
    --mode train --experiment supervised --domain embryology \
    --train_path         /path/to/splits/embryology/train.json \
    --val_path           /path/to/splits/embryology/val.json \
    --abstract_mesh_path /path/to/kg/embryology/splits/train_abstract_mesh.csv \
    --nodes_csv_path     /path/to/kg/embryology/nodes.csv \
    --unseen_codes_path  /path/to/splits/embryology/unseen_codes.json \
    --output_dir         /path/to/runs/embryology_tl_v9 \
    --label_mode unique_id --epochs 20 --warmup_epochs 5 \
    --batch_size 32 --lr 1e-3 --top_k 12 --fp16
```

---

## Evaluation

### Supervised
```bash
TOKENIZERS_PARALLELISM=false python train_pubmedbert_mesh_tl_v9.py \
    --mode test --experiment supervised --domain neurology \
    --test_path  /path/to/splits/neurology/test_supervised.json \
    --output_dir /path/to/runs/neurology_tl_v9
```


Replace `neurology` with `immunology` or `embryology` as needed.

---

## Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--warmup_epochs` | 5 | Epochs before translation activates. Classifier calibrates first. |
| `--top_k` | 0 | Local correction sparsity. 0 = all labels active. Embryology/immunology: try 12. |
| `--gamma_init` | -2.3 | Log-space init for correction scale γ. softplus(-2.3) ≈ 0.1. |
| `--lr` | 1e-3 | Learning rate. Best found from v8b sweep. |
| `--batch_size` | 32 | Batch size. |
| `--epochs` | 20 | Total training epochs. |
| `--fp16` | False | Mixed precision. Recommended on A30. |

---

## Output Files

All outputs are saved to `--output_dir`:

| File | Description |
|---|---|
| `best_model.pt` | Best checkpoint by validation nDCG@5 |
| `label_maps.json` | label index maps |
| `directions.npy` | Spurious direction matrix V [num_seen, 768] |
| `test_results_supervised.json` | Supervised evaluation metrics |
| `test_results_tail.json` | Tail-label evaluation metrics |

---

## Metrics Reported

- **nDCG@1,3,5** — primary metric, used for model selection
- **P@1, P@3, P@5** — precision at k
- **MRR** — mean reciprocal rank
- **mAP** — mean average precision
- **micro_F1** — micro-averaged F1 at threshold 0.5
- **micro_AUROC** — micro-averaged AUROC
- **Hits@10** — recall at 10

---

## Label Modes

| Mode | Description |
|---|---|
| `unique_id` | MeSH unique ID (default, recommended) |
| `heading` | MeSH heading string |
| `tree_numbers` | MeSH tree number(s) |
| `combined` | unique_id \|\| tree_number |

---

## Environment

- Python 3.9+
- CUDA-compatible GPU (tested on A30, 24GB)
- Conda environment: `bert_gzsl`
