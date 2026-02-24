# Multi-Label Narrative & Subnarrative Classification (BERT)

Multi-label classification for **Narrative** and **Subnarrative** labels using a BERT encoder (`bert-base-multilingual-cased`) with:

- **Hierarchical conditioning**: subnarrative head uses narrative logits as additional input  
- **Hierarchical consistency loss**: encourages predicted subnarratives to align with the active narrative  
- **Focal loss + pos_weight**: handles class imbalance  
- **Oversampling** with `WeightedRandomSampler`  
- Separate scripts for **training**, **inference**, and **evaluation**

> **Data is not included** in this repo. Place files in the structure below.

---

## Project Structure

```text
.
├── training.py
├── inference.py
├── evaluation.py
├── annotations/
│   └── annotation.txt
├── articles/
│   └── <article_id files...>
├── validation/
│   └── <article_id files...>
└── final_model/              # created by training.py
    ├── config.json
    ├── pytorch_model.bin     # or model.safetensors (optional)
    ├── tokenizer files...
    ├── narrative_mapping.json
    └── subnarrative_mapping.json
````

---

## Annotation Format (`annotations/annotation.txt`)

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
pip install torch transformers scikit-learn pandas numpy

# optional (only if you save/load safetensors):
pip install safetensors
```

GPU is optional. The code will automatically use CUDA if available.

---

## How to Run

### 1) Train the model

```bash
python training.py
```

This will:

* Read `annotations/annotation.txt`
* Load article texts from `articles/`
* Train with evaluation each epoch
* Save the final model and label mappings to `./final_model/`

Outputs:

* `final_model/` (model weights + tokenizer + mappings)
* `output/` (trainer checkpoints/logs)

---

### 2) Run inference (create submission file)

Put dev/validation articles in:

```text
validation/
```

Run:

```bash
python inference.py
```

This will:

* Load model + tokenizer from `./final_model/`
* Predict labels for each file in `validation/`
* Enforce hierarchical consistency on subnarratives
* Write predictions to:

Output:

* `submission.txt` (tab-separated: `article_id  narrative_labels  subnarrative_labels`)

---

### 3) Evaluate predictions

You have two options:

#### Option A (quick): edit paths inside `evaluation.py`

At the bottom of `evaluation.py`, update:

```python
gold_file = "annotations/annotation.txt"
pred_file = "submission.txt"
```

Then run:

```bash
python evaluation.py
```

#### Option B (recommended): wrapper script (no file edits)

Create `run_eval.py`:

```python
from evaluation import evaluate_files

evaluate_files("annotations/annotation.txt", "submission.txt")
```

Run:

```bash
python run_eval.py
```

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

`inference.py` enforces the convention:

* If narrative is empty or only `Other` → set subnarrative to `Other`
* Otherwise, for each predicted narrative, ensure at least one matching subnarrative exists
  If not → append `Narrative: Other`

### Dynamic label selection

Inference uses thresholds + fallback:

* Pick labels above primary threshold
* If none, force top label and optionally add 2nd if above fallback threshold

---

## Common Issues

* **File not found**: ensure `annotations/annotation.txt` and article text files exist under `articles/` and `validation/`.
* **Mismatch in `article_id` names**: `article_id` is used as a file name directly.
* **Long texts**: model uses `max_length=512` with truncation.

---

## Author / Contribution

Implemented end-to-end by **Abdul Wahab Madni** (training + inference + evaluation).

---


