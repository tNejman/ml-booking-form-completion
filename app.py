import json
import random
import time
from datetime import datetime
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List

from model2 import BasePredictionModel, AdvancedPredictionModel, train_and_evaluate

app = FastAPI(title="Nocarz Offer Suggestion Service")
base_model = BasePredictionModel()
advanced_model = AdvancedPredictionModel()

class OfferRequest(BaseModel):
    model_version: str
    description: str

class OfferResponse(BaseModel):
    room_type: str
    property_type: str
    bathrooms_text: str
    bedrooms: float  # specjalne przypadki np. 1.5
    beds: float
    accommodates: float
    amenities: List[str]
    model_version: str

def log_prediction(request_data: str, response_data: dict, model_name: str, processing_time: float):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input_length": len(request_data),
        "model_used": model_name,
        "prediction": response_data,
        "processing_time_ms": round(processing_time * 1000, 2)
    }
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

@app.post("/suggest_attributes", response_model=OfferResponse)
async def suggest_attributes(offer: OfferRequest):
    """
    Główny endpoint serwujący predykcje.
    Realizuje wymóg: "wybór modelu z perspektywy klienta powinien być przezroczysty" 
    """
    start_time = time.time()
    
    model_name = offer.model_version
    if model_name == "baseline":
        result = base_model.predict(offer.description)
    elif model_name == "advanced":
        result = advanced_model.predict(offer.description)
    else:
        result = {"Api error": f"unexptected model_version: {model_name}"}
    
    response_payload = {
        **result,
        "model_version": model_name
    }
    
    log_prediction(offer.description, result, model_name, time.time() - start_time)
    
    return response_payload

@app.get("/health")
def health_check():
    return {"status": "ok"}

# uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn

    LOG_FILE = "ab_test_logs.jsonl"
    
    train_and_evaluate(base_model=base_model, advanced_model=advanced_model)
    
    print("Start serwera Nocarz Suggestion Service...")
    uvicorn.run(app, host="0.0.0.0", port=8080)