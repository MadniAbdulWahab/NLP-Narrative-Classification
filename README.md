# Multi-Label Narrative & Subnarrative Classification (BERT)

Multi-label classification for **Narrative** and **Subnarrative** labels using a BERT encoder (`bert-base-multilingual-cased`) with:

- **Hierarchical conditioning**: subnarrative head uses narrative logits as additional input
- **Hierarchical consistency loss**: encourages predicted subnarratives to align with the active narrative
- **Focal loss + pos_weight**: handles class imbalance
- **Oversampling** with `WeightedRandomSampler`
- Separate scripts for **training**, **inference**, and **evaluation**

> **Data is not included** in this repo. Put your local files under `data/` as shown below.

---

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── train.py
│   ├── infer.py
│   └── eval.py
├── src/
│   ├── training.py
│   ├── inference.py
│   └── evaluation.py
├── data/                         # not committed (placeholder folders via .gitkeep)
│   ├── annotations/
│   │   └── annotation.txt
│   ├── articles/
│   │   └── <article_id files...>
│   └── validation/
│       └── <article_id files...>
├── models/                       # created by training (ignored unless using LFS)
│   └── final_model/
│       ├── config.json
│       ├── pytorch_model.bin     # or model.safetensors (optional)
│       ├── tokenizer files...
│       ├── narrative_mapping.json
│       └── subnarrative_mapping.json
└── outputs/                      # predictions + logs (not committed)
    ├── submission.txt
    └── output/                   # trainer checkpoints/logs
````

---

## Annotation Format (`data/annotations/annotation.txt`)

Tab-separated with **3 columns**:

```text
article_id<TAB>narrative_labels<TAB>subnarrative_labels
```

Rules:

* Multiple labels are separated by `;`
* Subnarratives follow `Narrative: Subnarrative` format
  Example: `Economy: Inflation`

---

## Setup

### 1) Create environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

GPU is optional. The code will automatically use CUDA if available.

---

## How to Run

> Run via the wrapper scripts in `scripts/` (recommended).

### 1) Train the model

```bash
python scripts/train.py
```

This will:

* Read `data/annotations/annotation.txt`
* Load article texts from `data/articles/`
* Train with evaluation each epoch
* Save the final model and label mappings to `models/final_model/`

Outputs:

* `models/final_model/` (model weights + tokenizer + mappings)
* `outputs/output/` (trainer checkpoints/logs)

---

### 2) Run inference (create submission file)

Put dev/validation articles in:

```text
data/validation/
```

Run:

```bash
python scripts/infer.py
```

This will:

* Load model + tokenizer from `models/final_model/`
* Predict labels for each file in `data/validation/`
* Enforce hierarchical consistency on subnarratives

Output:

* `outputs/submission.txt` (tab-separated: `article_id  narrative_labels  subnarrative_labels`)

---

### 3) Evaluate predictions

```bash
python scripts/eval.py
```

This evaluates:

* gold: `data/annotations/annotation.txt`
* predictions: `outputs/submission.txt`

---

## Metrics

Metrics printed:

* Averaged sample F1 for:

  * `(narrative:subnarrative)` pairs
  * narrative-only
  * subnarrative-only
* Macro F1 for:

  * narrative-only
  * subnarrative-only

---

## Notes on the Model

### Hierarchical conditioning

The model predicts `narrative_logits` first, then concatenates them with the pooled BERT output to predict `subnarrative_logits`:

* Narrative head: `BERT -> narrative_logits`
* Subnarrative head: `concat(BERT_pooled, narrative_logits) -> subnarrative_logits`

### Consistency enforcement at inference

`inference.py` enforces:

* If narrative is empty or only `Other` → set subnarrative to `Other`
* Otherwise, for each predicted narrative, ensure at least one matching subnarrative exists
  If not → append `Narrative: Other`

### Dynamic label selection

Inference uses thresholds + fallback:

* Pick labels above primary threshold
* If none, force top label and optionally add a 2nd if above a fallback threshold

---

## Common Issues

* **File not found**: ensure `data/annotations/annotation.txt` and article text files exist under `data/articles/` and `data/validation/`.
* **Mismatch in `article_id` names**: `article_id` is used as a file name directly.
* **Long texts**: model uses `max_length=512` with truncation.

---

## Author / Contribution

Implemented end-to-end by **Abdul Wahab Madni** (training + inference + evaluation).

```


