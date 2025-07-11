from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

import mlflow
import joblib
import pandas as pd
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import dagshub
from src.logging import logger
from text_prettifier import TextPrettifier
import uvicorn
import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
def clean_text(text):
    try:
        prettifier = TextPrettifier()
        text = text.lower()
        text = prettifier.remove_contractions(text)
        text = prettifier.remove_emojis(text)
        text = prettifier.remove_html_tags(text)
        text = prettifier.remove_urls(text)
        text = prettifier.remove_special_chars(text)
        text = prettifier.remove_stopwords(text)
        text = prettifier.remove_numbers(text)
        return text
    except Exception as e:
        logger.info(f"Error cleaning text: {e}")

mlflow.set_tracking_uri('https://dagshub.com/AyushAI14/Mlops-Capstone-Project.mlflow')
dagshub.init(repo_owner='AyushAI14', repo_name='Mlops-Capstone-Project', mlflow=True)

app = FastAPI()

# Metrics
registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total number of requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Latency of requests", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count", "Prediction count by class", ["prediction"], registry=registry)

# Load model and vectorizer
model = joblib.load('models/clfLR.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.get("/")
async def home(request: Request):
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = templates.TemplateResponse("index.html", {"request": request})
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.post("/predict")
async def predict(request: Request, text: str = Form(...)):
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

    raw_pred = model.predict(features_df)[0]
    prediction = "✨ Positive" if raw_pred == 1 else "👎 Negative"
    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
    return templates.TemplateResponse("index.html", {
    "request": request,
    "result": prediction,
    "raw_pred": raw_pred  # <-- Add this
    })

@app.get("/metrics")
async def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=5000)
