from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import NLUPredictor

# ---------------- Pydantic schemas ----------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Kullanıcı cümlesi")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=64)


class EntitySpan(BaseModel):
    type: str
    text: str
    start: int
    end: int


class PredictResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    entities: List[EntitySpan]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    count: int


# ---------------- Lifespan: modeli bir kez yükle ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Loading NLU model...")
    app.state.predictor = NLUPredictor()
    print(f"[API] Model ready on device: {app.state.predictor.device}")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Turkish Joint NLU API",
    description="Joint intent classification + slot filling for Turkish (BERTurk).",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------- Endpoints ----------------

@app.get("/", tags=["health"])
def root():
    return {
        "service": "Turkish Joint NLU API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health():
    predictor = getattr(app.state, "predictor", None)
    return {
        "status": "ok" if predictor is not None else "loading",
        "device": str(predictor.device) if predictor else None,
        "num_intents": len(predictor.intent_to_id) if predictor else None,
        "num_slots": len(predictor.slot_to_id) if predictor else None,
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    try:
        result = predictor.predict(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(request: BatchPredictRequest):
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    try:
        results = predictor.predict_batch(request.texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"results": results, "count": len(results)}