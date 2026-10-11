# Local API examples

Start from the repository root with the checkpoint in
`checkpoints/best_joint_nlu_model.pt`:

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs). Select an endpoint,
choose **Try it out**, enter the JSON body and choose **Execute**.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Load status, device and label counts |
| POST | `/predict` | One sentence |
| POST | `/predict/batch` | Up to 64 sentences |

## Single sentence

For `/predict`, use:

```json
{"text": "yarın sabah yediye alarm kurar mısın"}
```

An **illustrative** response, not a newly measured prediction:

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

`intent` is the predicted request type. `confidence` is an uncalibrated intent
softmax score. `entities` contains extracted spans; end offsets are exclusive,
so `text[start:end]` retrieves the span.

Other questions to try: `müzik aç`, `bugün istanbulda hava nasıl olacak`, and
`Taksim'de gezebileceğim önemli mekanlar var mı`. Predictions depend on the
supplied checkpoint.

## Batch and Python request

For `/predict/batch`, use:

```json
{"texts": ["hava nasıl", "müzik aç", "alarm kur", "istanbulda ne yenir"]}
```

The response contains `results` (one prediction object per sentence) and `count`.
A Python client can call the single endpoint directly:

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "yarın sabah yediye alarm kurar mısın"},
    timeout=10,
)
response.raise_for_status()
print(response.json())
```

## Input and loading limits

- A single request accepts 1–512 characters; empty text returns HTTP 422.
- Batches accept 1–64 items. Their items do not share the single-request length
  rule. Both prediction paths truncate to 64 tokenizer tokens.
- Returned `text` is the original input; there is no truncation flag.
- `/health` reports the device, 60 intent labels and 111 BIO labels. The latter
  represent 55 raw slot types, rather than 111 different entity types.
- Missing checkpoint, config or encoder/tokenizer assets prevent startup.
  An unavailable predictor returns 503; inference exceptions return 500.
- The service's OpenAPI version identifies the API, not the trained weights.

The API is a local demo. Terminal prediction uses the older decoder and a
different checkpoint path; see the README before comparing their entity spans.