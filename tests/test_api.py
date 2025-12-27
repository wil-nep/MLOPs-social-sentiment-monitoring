import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_sentiment_structure():
    payload = {"text": "Questo prodotto è fantastico!"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "label" in data
    assert "score" in data
    assert isinstance(data["label"], str)
    assert isinstance(data["score"], float)
