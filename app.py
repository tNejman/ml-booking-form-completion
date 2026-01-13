import json
import random
import time
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from model2 import AdvancedPredictionModel, BasePredictionModel, train_and_evaluate

app = FastAPI(
    title="Nocarz Offer Suggestion Service",
    description="API for suggesting form fields.",
    version="1.0.0",
)

base_model = BasePredictionModel()
advanced_model = AdvancedPredictionModel()
PREDICTION_LOG_FILE = "ab_test_logs.jsonl"
FEEDBACK_LOG_FILE = "feedback_logs.jsonl"


class OfferRequest(BaseModel):
    description: str


class OfferResponse(BaseModel):
    prediction_id: str
    room_type: Optional[str] = None
    property_type: Optional[str] = None
    bathrooms_text: Optional[str] = None
    bedrooms: Optional[float] = None
    beds: Optional[float] = None
    accommodates: Optional[float] = None
    amenities: List[str] = []
    model_version: str


class FeedbackRequest(BaseModel):
    prediction_id: str
    room_type: Optional[str] = None
    property_type: Optional[str] = None
    bathrooms_text: Optional[str] = None
    bedrooms: Optional[float] = None
    beds: Optional[float] = None
    accommodates: Optional[float] = None
    amenities: List[str] = []


def log_prediction(
    pred_id: str, description: str, result: dict, model_name: str, duration: float
):
    log_entry = {
        "prediction_id": pred_id,
        "timestamp": datetime.now().isoformat(),
        "model_used": model_name,
        "input_length": len(description),
        "prediction": result,
        "processing_time_ms": round(duration * 1000, 2),
    }
    with open(PREDICTION_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def _prepare_response(result_dict: dict, model_ver: str, pred_id: str) -> dict:
    if "amenities" not in result_dict or result_dict["amenities"] is None:
        result_dict["amenities"] = []

    return {**result_dict, "model_version": model_ver, "prediction_id": pred_id}


@app.post("/predict/baseline", response_model=OfferResponse)
async def predict_baseline(offer: OfferRequest):
    """
    Uses baseline model.
    """
    prediction_id = str(uuid.uuid4())
    result = base_model.predict(offer.description)
    return _prepare_response(result, "baseline_forced", prediction_id)


@app.post("/predict/advanced", response_model=OfferResponse)
async def predict_advanced(offer: OfferRequest):
    """
    Uses advanced ML model.
    """
    prediction_id = str(uuid.uuid4())
    result = advanced_model.predict(offer.description)
    return _prepare_response(result, "advanced_forced", prediction_id)


@app.post("/predict/ab_test", response_model=OfferResponse)
async def predict_ab_test(offer: OfferRequest):
    """
    Randomly selects model (50/50).
    It saves results into logs.
    """
    start_time = time.time()
    prediction_id = str(uuid.uuid4())
    start_time = time.time()

    if random.random() < 0.5:
        model = base_model
        model_name = "baseline"
    else:
        model = advanced_model
        model_name = "advanced"

    result = model.predict(offer.description)

    duration = time.time() - start_time
    log_prediction(prediction_id, offer.description, result, model_name, duration)

    return _prepare_response(result, model_name, prediction_id)


@app.post("/feedback")
async def save_feedback(feedback: FeedbackRequest):
    """
    Gets info about what user finally sent.
    """
    log_entry = feedback.dict()
    log_entry["timestamp"] = datetime.now().isoformat()

    with open(FEEDBACK_LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"status": "feedback_saved", "id": feedback.prediction_id}


@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    train_and_evaluate(base_model=base_model, advanced_model=advanced_model)

    print("\n=================================================")
    print(f" LOGS of A/B experiment go to: {PREDICTION_LOG_FILE}")
    print(f" LOGS of feedbacks go to: {FEEDBACK_LOG_FILE}")
    print(" Swagger UI: http://localhost:8080/docs")
    print("=================================================\n")

    uvicorn.run(app, host="0.0.0.0", port=8080)
