import os
import re
import joblib
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

# !! ID !! 
RUN_ID = "a1b66d369f734a7f9923326cfbb8d6cf" 
ARTIFACT_PATH = "distilbert-sentiment-model"
MODEL_URI = f"runs:/{RUN_ID}/{ARTIFACT_PATH}"

print(f"Downloading model artifacts from: {MODEL_URI}...")
local_model_dir = mlflow.artifacts.download_artifacts(MODEL_URI)
print(f"Artifacts downloaded to: {local_model_dir}")

tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("Model pipeline created successfully.")
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to predict the sentiment of a given text review.",
    version="1.0.0"
)


class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str = Field(..., description="The predicted label, e.g., 'POSITIVE' or 'NEGATIVE'.")
    score: float = Field(..., ge=0, le=1, description="The confidence score of the prediction.")

def clean_text(text: str) -> str:
    """Очищает текст, оставляя только буквы и пробелы в нижнем регистре."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


@app.get("/")
def read_root():
    """Корневой эндпоинт для проверки работоспособности API."""
    return {"message": "Welcome to the Sentiment Analysis API!"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    result = sentiment_pipeline(request.text)[0]
    return result