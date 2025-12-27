from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Social Sentiment Monitoring API")


class TextIn(BaseModel):
    text: str


class SentimentOut(BaseModel):
    label: str
    score: float


sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentOut)
def predict_sentiment(payload: TextIn):
    result = sentiment_pipeline(payload.text)[0]
    return SentimentOut(label=result["label"], score=float(result["score"]))
