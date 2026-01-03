# MLOps Social Sentiment Monitoring

Public demo project for monitoring a company's **online reputation** using social‑media sentiment analysis.  
The repository implements a sentiment API, CI/CD pipeline, basic MLOps practices, and monitoring with Prometheus/Grafana.

# MLOps Social Sentiment Monitoring

Public demo project for monitoring a company's **online reputation** using social‑media sentiment analysis.  
The repository implements a sentiment API, CI/CD pipeline, basic MLOps practices, and monitoring with Prometheus/Grafana.

## Prerequisites
- Python 3.11+
- Docker and Docker Compose
- GitHub account for CI/CD

## Repository Structure

- `app/`
  - FastAPI application (endpoints, models, configuration)
  - Sentiment inference code using a Hugging Face model
- `training/`
  - Reference scripts demonstrating fine-tuning capability
  - Project uses pretrained model as per requirements
- `tests/`
  - pytest test suite (unit + integration tests for API and model)
- `monitoring/`
  - Prometheus and Grafana configuration
  - Additional assets for observability dashboards
- Root files
	- `Dockerfile` – FastAPI service container image
  - `docker-compose.yml` – API + Prometheus + Grafana stack
  - `prometheus.yml` – Prometheus configuration
  - `requirements.txt` – Python dependencies
  - `.github/workflows/` – CI/CD pipeline (GitHub Actions)
  - `.gitignore`, `.dockerignore` – Git and Docker ignore rules


## Main Features

- **Sentiment API**
  - REST endpoint `/predict` to classify text as positive / negative / neutral.
  - Health‑check endpoint `/health` for monitoring and readiness.
  - `/metrics`  endpoint exposing Prometheus metrics from the FastAPI 
- **Model & Data**
  - Pretrained Hugging Face model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Specialized for Twitter/social media sentiment analysis
  - Classifies text as positive, negative, or neutral
  - Training scripts included for reference (fine-tuning demonstration)
- **Testing**
  - pytest tests for:
    - Core utility functions (preprocessing, validation).
    - FastAPI routes (status codes and response schema).
    - Basic “sanity” checks on model predictions.
- **CI/CD**
  - GitHub Actions workflow:
    - Run tests on every push/PR
    - Build and test Docker image
    - Optional Docker Hub push
- **Monitoring**
  - Prometheus metrics exposed from the FastAPI app (`/metrics`).
  - Grafana dashboards for:
    - Request rate, latency, error rate.
    - Distribution of sentiment predictions over time.

## Dockerized Service & Monitoring

## How This Repository Will Be Used

1. **Local development**
   - Clone the repo, create a virtual environment, install `requirements.txt`
   - Run the FastAPI app locally: `uvicorn app.main:app --reload`
   - Execute tests: `pytest tests/ -v`
2. **Dockerized API service**
   - Build and run: `docker compose up -d --build`
   - Access endpoints:
     - `http://localhost:8000/health` – health check
     - `http://localhost:8000/docs` – interactive API docs
     - `http://localhost:8000/metrics` – Prometheus metrics
   - Stop: `docker compose down`
3. **Monitoring (Prometheus + Grafana)**
   - Start the monitoring stack using `docker-compose.monitoring.yml` and `prometheus.yml`.
   - Services exposed:
     - Prometheus at `http://localhost:9090`
     - Grafana at `http://localhost:3000` (default admin/admin123, configurable in the compose file).
   - Use Grafana dashboards to explore:
     - request rate, latency and error rate;
     - distribution of sentiment predictions over time.
   - Stop the monitoring stack.
4. **Model retraining**
   - Add or update training data in the `training/` pipeline.
   - Run the training script to generate new model artifacts.
   - Commit/push changes so the CI/CD pipeline can test and redeploy the updated model.
5. **Deploy to Hugging Face Spaces**
   - Create a Space on Hugging Face (e.g., for Gradio interface).
   - Use the CI/CD pipeline to push the model or app automatically (requires HF_TOKEN secret).
   - Manual deploy: `huggingface-cli upload model/fine_tuned_model --repo-id your-username/your-space`
6. **Docker / environment limitations(Codespaces)**
   - The project has been developed in a constrained environment (e.g. GitHub Codespaces with ~30 GB of disk space).
   - To avoid  `No space left on device`  errors when building the Docker image, the API container is intentionally lightweight:
     - **PyTorch CPU-only** (~200MB) is installed instead of PyTorch with CUDA (~2GB+), since Codespaces has no GPU.
     - The Docker container is used **only for inference** (API predictions), not for training.
     - **Model training** is done on Google Colab (with free GPU), and the trained model is then uploaded to this repo.
     - This design demonstrates a complete MLOps setup (API + CI/CD + monitoring) while keeping the Docker image small enough for Codespaces.
   - **Workflow**: Train on Colab with GPU → Export model → Upload to repo → Deploy API with Docker (CPU inference only).
  

