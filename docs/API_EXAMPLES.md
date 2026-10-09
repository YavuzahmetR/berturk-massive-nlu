# API usage examples

These request/response illustrations were retained from the original README. They are not fresh inference results or guarantees of exact predictions. The artist example's character end offset and BIO label count were corrected to match the literal example/mapping. Runtime response semantics are unchanged. `num_slots` counts BIO classifier labels (111), not raw slot types (55).

### 4. REST API

```bash
uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
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
  "num_slots": 111
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
  -d '{"text": "rabiask hayranıyım bu şarkıdan sonra onun şarkılarından çal"}'
```

**Response:**

```json
{
  "text": "rabiask hayranıyım bu şarkıdan sonra onun şarkılarından çal",
  "intent": "play_music",
  "confidence": 0.9959,
  "entities": [
    {"type": "artist_name", "text": "rabiask", "start": 0, "end": 7}
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
