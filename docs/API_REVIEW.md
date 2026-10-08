# API contract and verification

Updated: **2026-10-09**.

## Setup and example

From the repository root, install with `python -m pip install -e "."`. Supply `checkpoints/best_joint_nlu_model.pt`; the trained checkpoint is not included. `configs/joint_bert.yaml` and both label mappings must match the checkpoint. The BERTurk tokenizer and initial encoder require Hugging Face access or a matching local cache.

```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/docs`. Example request:

```json
{"text": "yarın sabah yediye alarm kurar mısın"}
```

Observed response with the checkpoint used in CPU verification:

```json
{
  "text": "yarın sabah yediye alarm kurar mısın",
  "intent": "alarm_set",
  "confidence": 0.992397129535675,
  "entities": [
    {"type": "date", "text": "yarın", "start": 0, "end": 5},
    {"type": "time", "text": "sabah yediye", "start": 6, "end": 18}
  ]
}
```

Checkpoint SHA-256: `daf393caece62ca84756ed09059c04a12f16573318e0faad42006010597a8ff6`.
Different checkpoints or environments may produce different values. Entity end offsets are exclusive: `text[start:end]`. Confidence is intent softmax output, not calibrated confidence.

## Request and response contract

| Request or condition | Behavior |
|---|---|
| `GET /` | Service name, status and docs path |
| `GET /health` | `status`, `device`, `num_intents`, `num_slots` |
| `GET /docs` | Interactive Swagger UI |
| `POST /predict` | `text`, `intent`, `confidence`, `entities` |
| `POST /predict/batch` | Accepts `texts`; returns `results` and `count` |
| Single text empty, missing, numeric or longer than 512 characters | HTTP 422 |
| Empty batch or more than 64 items | HTTP 422 |
| Predictor object absent | Prediction HTTP 503 |
| Inference exception | Existing HTTP 500 path |

Loaded CPU health response:

```json
{"status": "ok", "device": "cpu", "num_intents": 60, "num_slots": 111}
```

The 111 BIO classifier labels differ from the 55 raw slot types. Batch example:

```json
{"texts": ["müzik aç", "alarm kur"]}
```

## Recorded checks

Seventeen bounded HTTP calls had identical status and JSON before/after cleanup using the same checkpoint: service information, health/Swagger UI, three single predictions, a normal batch, six invalid requests, empty/long batch items and predictor-absent behavior. The last branch was invoked without lifespan startup; it does not establish successful startup with a missing checkpoint. See [CLEANUP_RESULTS.json](CLEANUP_RESULTS.json).

Inference, model, Dataset, training, tuning, evaluation and their command modules imported after namespace cleanup. Setuptools discovered `src`, its subpackages, `scripts` and `api`. Three preparation imports were not run because the verification environment lacked `datasets`; the dependency remains declared in both installation manifests. Data preparation and training were not rerun. Scientific test data, mappings, configuration and historical reports retained their bytes.

## Known limitations

| Topic | Observation |
|---|---|
| Checkpoint access | No trained weights or published download link are included. Installation alone is insufficient for inference. |
| Startup | Checkpoint/config/cache errors during predictor construction fail lifespan startup. |
| API/CLI paths | API uses `checkpoints/best_joint_nlu_model.pt`; CLI uses the original root-level path. Comparisons require the same checkpoint bytes. |
| Batch validation | Items do not share the single-request 1–512-character rule. Empty and 540-character batch items returned HTTP 200; analogous single requests returned 422. |
| Truncation | Tokenization truncates to 64 tokens. Responses retain the original text without reporting truncation. Character and token limits differ. |
| Entity decoding | API subtoken-merging behavior differs from CLI decoding. Equal entity boundaries are not guaranteed across those interfaces. |
| Model identity | Prediction responses lack a checkpoint hash/version field. OpenAPI version `1.0.0` identifies the service, not the checkpoint. |
| Evaluation output | The scientific evaluator prints results to the terminal; a common JSON release exporter is not included. |

The cleanup did not change HTTP fields, decoders, models, optimizer settings, splits or metric formulas.
