# Turkish Joint NLU with BERTurk

## Overview

Predict the **intent and entity spans** of a Turkish sentence with one shared
BERTurk encoder. For example, an alarm request can produce `alarm_set` together
with its date and time. This is a supervised model study with terminal inference
and a local FastAPI demo.

## How it works

```text
Turkish sentence -> BERTurk tokenizer -> shared BERTurk encoder
                                      -> intent classifier
                                      -> token classifier -> entity spans
```

The intent classifier reads the first token's hidden state. The slot classifier
labels each token using BIO notation: beginning, inside or outside an entity.
Training combines intent cross-entropy with slot cross-entropy, weighted by 1.5.
Padding labels are ignored. The encoder has 12 layers and 768 hidden dimensions;
it is fine-tuned together with both classifiers.

## Data and evaluation

The Turkish `tr-TR` portion of **Amazon MASSIVE** supplies the annotated text.
The official partitions are retained: **11,514 train**, **2,033 validation** and
**2,974 test** sentences. There are **60 intents** and **55 slot types**, encoded
as **111 BIO labels**. Preparation checks row counts and overlapping row IDs.

Normal training selects the lowest validation loss. Final intent results use
weighted-F1; slot results use entity-level F1 with `seqeval`. These are different
measurements, and neither is the same as token accuracy.

## Recorded results

| Model | Reported split | Intent weighted-F1 | Slot entity F1 |
|---|---|---:|---:|
| Earlier v1 reference | Test | 87.54% | 73.94% |
| v3 | Test | 87.99% | 74.16% |
| Best recorded Optuna trial | Validation | — | 68.75% |

The v3 changes were a larger slot-loss weight, longer warmup and slightly more
dropout. They changed together; the small gains cannot be attributed to one
setting. The Optuna validation score cannot be compared directly with test F1.

These are historical documented results. Complete evaluation logs are unavailable,
and a separately supplied checkpoint has not been linked to these scores.
[Experiment details](reports/HISTORICAL_RESULTS.md) retain the additional metrics,
settings and trial observations.

## Project layout

| Path | Purpose |
|---|---|
| `src/data/` | Text preparation, token alignment and dataset loading |
| `src/modeling/` | Shared encoder, two classifiers and loss |
| `src/training/`, `src/evaluation/` | Training, optional search and metrics |
| `src/inference/`, `api/app.py` | Terminal prediction and local API |
| `scripts/` | Commands that call the implementation |
| `configs/joint_bert.yaml` | Model and training settings |
| `data/raw/`, `data/processed/` | Official partitions and label-to-ID mappings |
| `reports/`, `docs/` | Recorded results and API examples |

The mapping JSON files translate model output positions into label names.
The YAML file holds settings; neither file is an additional model.

## Setup

Python **3.10+** is required. From the repository root:

```bash
python -m venv .venv
# Activate: Windows .venv\Scripts\Activate.ps1; macOS/Linux source .venv/bin/activate
python -m pip install -r requirements.txt
```

Use a PyTorch build compatible with your device. Model/tokenizer access or a
matching local cache is needed for `dbmdz/bert-base-turkish-cased`. Dependencies
are not locked to the original training environment.

The checkpoint is **not included** in Git. The API, training and evaluation use
`checkpoints/best_joint_nlu_model.pt`; the original terminal command instead uses
`best_joint_nlu_model.pt` in the repository root. Supply the same weights at the
path required by each command. Training again will not recreate identical weights.

## Usage

```bash
python -m scripts.predict "yarın sabah yediye alarm kurar mısın"
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) to try `/predict` or
`/predict/batch`. Requests, illustrative outputs and input limits are in
[API_EXAMPLES.md](docs/API_EXAMPLES.md).

Training and evaluation are separate commands:

```bash
python -m scripts.train --config configs/joint_bert.yaml
python -m scripts.evaluate_model
python -m scripts.tune
```

Optuna is optional and runs ten trials. Data preparation is only needed when
rebuilding inputs: `python -m scripts.prepare_data`, then
`python -m scripts.generate_slot_mapping`. These commands can overwrite the
supplied partitions and mappings. Run all commands from the repository root.
The older misspelled evaluation command remains a compatibility entry point.

## Limitations

- Fixed vocabulary: unfamiliar requests still receive one of the known intents.
- Inputs are truncated at 64 tokenizer tokens. Intent confidence is an
  uncalibrated softmax score, not a guarantee of correctness.
- API and terminal BIO decoding differ, so some consecutive subword spans differ.
- The test split has already been inspected; reuse is historical re-evaluation.
- No uncertainty estimate, arbitrary-input robustness or production deployment
  claim is made. Missing model assets prevent API startup.

Data: Amazon MASSIVE. Encoder: BERTurk (`dbmdz`). Main libraries: PyTorch,
Transformers, `seqeval`, Optuna and FastAPI. No repository license file is present.