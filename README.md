# MLOps Social Sentiment Monitoring

Public demo project for monitoring a company's **online reputation** using social‑media sentiment analysis.  
The repository implements a sentiment API, CI/CD pipeline, basic MLOps practices, and monitoring with Prometheus/Grafana.

## Repository Structure (planned)

- `app/`
  - FastAPI application (endpoints, models, configuration)
  - Sentiment inference code using a Hugging Face model
- `training/`
  - Scripts for (re)training / fine‑tuning the sentiment model
  - Utilities for dataset loading and preprocessing
- `tests/`
  - pytest test suite (unit + integration tests for API and model)
- `monitoring/`
  - Prometheus and Grafana configuration
  - `docker-compose.yml` to run API + Prometheus + Grafana locally
- `model/` (optional, ignored if large)
  - Saved model weights / artifacts used in production
- Root files
  - `Dockerfile` – container image for the FastAPI service
  - `requirements.txt` – Python dependencies
  - `.github/workflows/` – CI/CD pipelines (GitHub Actions)
  - `.gitignore` – ignore virtualenvs, caches, logs, etc.

## Main Features

- **Sentiment API**
  - REST endpoint `/predict` to classify text as positive / negative / neutral.
  - Health‑check endpoint `/health` for monitoring and readiness.
- **Model & Data**
  - Pretrained Hugging Face model (e.g. RoBERTa for Twitter sentiment).
  - Support for updating the dataset and re‑training the model.
- **Testing**
  - pytest tests for:
    - Core utility functions (preprocessing, validation).
    - FastAPI routes (status codes and response schema).
    - Basic “sanity” checks on model predictions.
- **CI/CD**
  - GitHub Actions workflow to:
    - Install dependencies and run pytest on every push/PR.
    - Build Docker image for the API.
    - Optionally deploy to a Hugging Face Space or other environment.
- **Monitoring**
  - Prometheus metrics exposed from the FastAPI app (`/metrics`).
  - Grafana dashboards for:
    - Request rate, latency, error rate.
    - Distribution of sentiment predictions over time.

## How This Repository Will Be Used

1. **Local development**
   - Clone the repo, create a virtual environment, install `requirements.txt`.
   - Run the FastAPI app locally (e.g. with `uvicorn`) and execute tests with `pytest`.
2. **Dockerized service**
   - Build the Docker image using the provided `Dockerfile`.
   - Run the container alone or together with Prometheus/Grafana via `docker compose`.
3. **Monitoring**
   - Start the monitoring stack from `monitoring/docker-compose.yml`.
   - Explore metrics and sentiment trends via predefined Grafana dashboards.
4. **Model retraining**
   - Add or update training data in the `training/` pipeline.
   - Run the training script to generate new model artifacts.
   - Commit/push changes so the CI/CD pipeline can test and redeploy the updated model.

> Note: Detailed setup and run instructions (commands, environment variables, and configuration examples) will be added incrementally as the project evolves.
