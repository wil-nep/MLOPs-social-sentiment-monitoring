from fastapi import FastAPI

app = FastAPI(title="Social Sentiment Monitoring API")


@app.get("/health")
def health_check():
    return {"status": "ok"}
