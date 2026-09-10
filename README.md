#  Turkish Joint NLU

**Turkish Natural Language Understanding (NLU) system** — a production-ready, end-to-end project that performs both **intent classification** and **slot filling** in a single model.

Built on BERTurk, servable via FastAPI, tuned with Optuna.

---

## 📌 Table of Contents

- [Features](#-features)
- [Metrics](#-metrics)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Optuna Experience & Lessons Learned](#-optuna-experience--lessons-learned)
- [Technical Details](#-technical-details)
- [License](#-license)

---

## ✨ Features

-  **Joint Architecture** — Intent and slot predictions from a single BERTurk forward pass
-  **BERTurk** — `dbmdz/bert-base-turkish-cased` pre-trained model
-  **Comprehensive Evaluation** — Intent F1 + entity-level Slot F1 (seqeval)
-  **Leakage-free Pipeline** — Strict train / validation / test split
-  **Optimized Training** — AMP, early stopping, class weights support
-  **Optuna Tuning** — Automatic hyperparameter search pipeline
-  **FastAPI Server** — Single and batch inference endpoints, Swagger UI
-  **CLI + Interactive Mode** — Quick testing from the terminal


---

## 📊 Metrics

**Locked Test Split** (2,974 examples, fully unseen):

| Version | Intent F1 | Slot F1 | Slot Precision | Slot Recall |
|---|---|---|---|---|
| **v1** (baseline) | 87.54% | 73.94% | 72.25% | 75.70% |
| **v3** (final) ⭐ | **87.99%** | **74.16%** | **72.34%** | **76.06%** |
| Optuna Best Trial | — | 68.75% | — | — |

**Improvements in v3 over v1:**

| Change | Reason |
|---|---|
| `lambda_slot: 1.0 → 1.5` | More weight on slot loss, F1 ↑ |
| `warmup_ratio: 0.1 → 0.15` | BERT fine-tuning stability |
| `dropout_rate: 0.1 → 0.12` | Slight regularization |

**Gain:** Intent +0.45 points, Slot +0.22 points.

---

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │   Input Text        │
                    │   "yarın alarm kur" │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   BERTurk Tokenizer │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   BERTurk Encoder   │
                    │   (12-layer, 768d)  │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
     ┌──────────▼──────────┐      ┌───────────▼───────────┐
     │  [CLS] Pooler Head  │      │  Token Classification │
     │  ── Intent ──       │      │  ── Slot (BIO) ──     │
     │  Dropout + Linear   │      │  Dropout + Linear     │
     └─────────────────────┘      └───────────────────────┘
              │                              │
       60 intent classes              55 slot labels
       (CrossEntropy)                 (BIO CrossEntropy)
```

**Loss Function:**

```
Total Loss = Intent Loss + λ_slot × Slot Loss

λ_slot = 1.5 (more weight on slot)
```

---

## 📁 Project Structure

```
turkish-nlu/
├── api/
│   └── app.py                    # FastAPI server
├── checkpoints/
│   └── best_joint_nlu_model.pt   # Best trained checkpoint
├── configs/
│   └── joint_bert.yaml           # Training configuration
├── data/
│   ├── raw/                      # Raw JSONL splits
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── processed/                # Label mappings
│       ├── intent_to_id.json
│       └── slot_to_id.json
├── scripts/
│   ├── train.py                  # Training script
│   ├── evaluate_model.py         # Test evaluation
│   ├── predict.py                # CLI inference
│   ├── tune.py                   # Optuna hyperparameter search
│   └── generate_slot_mapping.py  # Slot mapping generator
├── src/
│   ├── data/
│   │   └── dataset.py            # PyTorch Dataset + Collate
│   ├── modeling/
│   │   ├── joint_nlu_model.py    # Main model + Parser + Encoder
│   │   └── losses.py             # Joint loss function
│   └── inference/
│       └── predictor.py          # Shared NLU predictor
├── tests/
│   ├── test_alignment.py         # Slot alignment test
│   └── test_pipeline.py          # End-to-end pipeline test
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### Requirements

- Python 3.10+
- CUDA 12.1+ (optional, for GPU)
- 4 GB+ VRAM (tested on RTX 3050 laptop)

### Steps

```bash
# 1) Clone the repo
git clone <repo-url>
cd turkish-nlu

# 2) Virtual environment
python -m venv venv
source venv/bin/activate         # Linux/Mac
# or
venv\Scripts\activate            # Windows

# 3) Dependencies
pip install -r requirements.txt

# 4) (For GPU) CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5) Install package (for import paths)
pip install -e .
```

---

## 💻 Usage

### 1. Training

```bash
python -m scripts.train
```

With a custom config:

```bash
python -m scripts.train --config configs/joint_bert.yaml
```

**Duration:** ~10-12 minutes (RTX 3050 laptop)

---

### 2. Evaluation

```bash
python -m scripts.evaluate_model
```

**Output:**

```
📊 FINAL LOCKED TEST METRICS — INTENT
  Accuracy  : 88.03%
  F1        : 87.99%  (weighted)

🏷️  FINAL LOCKED TEST METRICS — SLOT (entity-level)
  Precision : 72.34%
  Recall    : 76.06%
  F1        : 74.16%
  ── Token-level accuracy (baseline): 90.54%
```

---

### 3. CLI Inference

```bash
python -m scripts.predict "yarın sabah yediye alarm kurar mısın"
```

**Output:**

```
📝 Sentence: yarın sabah yediye alarm kurar mısın
------------------------------------------------------------
🎯 Intent : alarm_set  (confidence: 99.85%)
🏷️  Slots  :
     • date                 -> "yarın"          [0:5]
     • time                 -> "sabah yediye"   [6:18]
------------------------------------------------------------
```

Interactive mode:

```bash
python -m scripts.predict
```

---

### 4. REST API

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

**Swagger UI:** `http://localhost:8000/docs`

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| POST | `/predict` | Single-sentence inference |
| POST | `/predict/batch` | Batch inference (up to 64 texts) |

---

#### 🩺 `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "ok",
  "device": "cuda",
  "num_intents": 60,
  "num_slots": 55
}
```

---

#### 🎯 `POST /predict` — Single Sentence

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "yarın sabah yediye alarm kurar mısın"}'
```

**Response:**

```json
{
  "text": "yarın sabah yediye alarm kurar mısın",
  "intent": "alarm_set",
  "confidence": 0.9924,
  "entities": [
    {"type": "date", "text": "yarın", "start": 0, "end": 5},
    {"type": "time", "text": "sabah yediye", "start": 6, "end": 18}
  ]
}
```

---

**Request 2 — Music intent:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "rabia'\''nın hayranıyım bu şarkıdan sonra onun şarkılarından çal"}'
```

**Response:**

```json
{
  "text": "rabiask hayranıyım bu şarkıdan sonra onun şarkılarından çal",
  "intent": "play_music",
  "confidence": 0.9959,
  "entities": [
    {"type": "artist_name", "text": "rabiask", "start": 0, "end": 9}
  ]
}
```

---

**Request 3 — Weather query:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "bugün istanbulda hava nasıl olacak"}'
```

**Response:**

```json
{
  "text": "bugün istanbulda hava nasıl olacak",
  "intent": "weather_query",
  "confidence": 0.9990,
  "entities": [
    {"type": "date", "text": "bugün", "start": 0, "end": 5},
    {"type": "place_name", "text": "istanbulda", "start": 6, "end": 16}
  ]
}
```

---

**Request 4 — Location recommendation:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Taksim'\''de gezebileceğim önemli mekanlar var mı"}'
```

**Response:**

```json
{
  "text": "Taksim'de gezebileceğim önemli mekanlar var mı",
  "intent": "recommendation_locations",
  "confidence": 0.7059,
  "entities": [
    {"type": "place_name", "text": "Taksim'de", "start": 0, "end": 9}
  ]
}
```

---

#### 📦 `POST /predict/batch` — Batch Inference

**Request:**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "hava nasıl",
      "müzik aç",
      "alarm kur",
      "istanbulda ne yenir"
    ]
  }'
```

**Response:**

```json
{
  "results": [
    {
      "text": "hava nasıl",
      "intent": "weather_query",
      "confidence": 0.9984,
      "entities": []
    },
    {
      "text": "müzik aç",
      "intent": "play_music",
      "confidence": 0.9664,
      "entities": []
    },
    {
      "text": "alarm kur",
      "intent": "alarm_set",
      "confidence": 0.9979,
      "entities": []
    },
    {
      "text": "istanbulda ne yenir",
      "intent": "recommendation_locations",
      "confidence": 0.9253,
      "entities": [
        {"type": "place_name", "text": "istanbulda", "start": 0, "end": 10}
      ]
    }
  ],
  "count": 4
}
```

---

#### ⚠️ Error Response — Empty Input

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": ""}'
```

**Response (422 Unprocessable Entity):**

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "text"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

---

#### 🐍 Python Client Example

```python
import requests

API_URL = "http://localhost:8000"


def predict(text: str) -> dict:
    r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
    r.raise_for_status()
    return r.json()


def predict_batch(texts: list) -> list:
    r = requests.post(f"{API_URL}/predict/batch", json={"texts": texts}, timeout=30)
    r.raise_for_status()
    return r.json()["results"]


if __name__ == "__main__":
    # Single sentence
    result = predict("yarın sabah yediye alarm kurar mısın")
    print(f"Intent: {result['intent']} ({result['confidence']:.2%})")
    for ent in result["entities"]:
        print(f"  {ent['type']:<15s} -> {ent['text']!r}")

    # Batch
    print("\n--- Batch ---")
    for item in predict_batch(["hava nasıl", "müzik aç", "alarm kur"]):
        print(f"{item['text']:<15s} -> {item['intent']} ({item['confidence']:.2%})")
```

---

### 5. Hyperparameter Tuning (Optuna)

```bash
python -m scripts.tune
```

Runs 10 trials and saves the best config to `configs/joint_bert_best.yaml`. **Duration:** ~40-50 min.

---

## 🧪 Optuna Experience & Lessons Learned

We ran **10 trials** with Optuna:

| Trial | Slot F1 | Parameters |
|---|---|---|
| **Trial 0** | **0.6875** | lr=1.8e-5, dropout=0.29, CW=none |
| Trial 1 | 0.1793 | CW=both → **collapsed** |
| Trial 2 | 0.6404 | lr=1.4e-5, dropout=0.10 |
| Trial 3-6 | — | Pruned (early-stopped) |

**Optuna's best trial (0.6875) could not beat the manual baseline (0.7394).**

### Why?

1. **High dropout (0.29)** — Made it harder to learn slot boundaries (B-X, I-X)
2. **Label smoothing** — Smoothens intent but hurts slot
3. **Val loss ≠ Slot F1** — Optuna was guided by the wrong metric

### Takeaways

| Finding | Decision |
|---|---|
| ✅ Class weights collapse slot F1 | Keep disabled |
| ✅ High dropout hurts | 0.1-0.12 range is optimal |
| ✅ Label smoothing hurts slot | Keep disabled |
| ✅ Baseline (v1) is already strong | Light tuning is enough |

**A negative result is still a result.** The v3 config is the product of these lessons.

---

## 🛠️ Technical Details

### Model Architecture

| Layer | Detail |
|---|---|
| **Encoder** | BERTurk base cased (12-layer, 768 hidden) |
| **Intent Head** | `[CLS]` pooler → Dropout(0.12) → Linear(768, 60) |
| **Slot Head** | Token hidden → Dropout(0.12) → Linear(768, 55) |
| **Loss** | CE (intent) + 1.5 × CE (slot, ignore=-100) |
| **Optimizer** | AdamW (lr=3e-5, wd=0.01) |
| **Scheduler** | Linear warmup (15%) + linear decay |
| **Max Length** | 64 tokens |
| **Batch Size** | 16 |

### Slot Alignment

The Massive dataset marks slots with square brackets:

```
"şu an ciddi [artist_name : adele] hayranıyım"
```

`MassiveAnnotationParser`:
1. Removes the slot markers → clean text
2. Extracts character offsets → `[(5, 10, "artist_name")]`

`TokenAlignmentEncoder`:
1. Reads the tokenizer's `offset_mapping`
2. Aligns each slot span to tokens
3. Converts to BIO format: `B-artist_name`, `I-artist_name`, `O`
4. Uses `-100` (ignore index) for padding and special tokens

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- **Massive Dataset** — Amazon Alexa + Turkish community contribution
- **BERTurk** — `dbmdz` (Bekir Karahan, Stefan Schweter)
- **Hugging Face Transformers** — Model and tokenizer infrastructure
- **Optuna** — Hyperparameter search

---

