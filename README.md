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
- Hugging Face account for model hosting (optional)

## Repository Structure

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
  - Additional assets for observability dashboards.
- `model/` (optional, ignored if large)
  - Saved model weights / artifacts used in production
- Root files
	- `Dockerfile`  – container image for the FastAPI service.
  - `docker-compose.yml`  – compose file for the API container.
  - `docker-compose.monitoring.yml`  – compose file for Prometheus + Grafana stack.
  - `prometheus.yml`  – Prometheus scrape configuration for the FastAPI app.
  - `requirements.txt`  – Python dependencies (API‑oriented, lightweight).
  -  `.github/workflows/`  – CI/CD pipelines (GitHub Actions).
  -  `.gitignore`  – ignore virtualenvs, caches, logs, etc.
  -  `.dockerignore`  – exclude heavy folders (training data, models, venv, etc.) from the Docker build context.


## Main Features

- **Sentiment API**
  - REST endpoint `/predict` to classify text as positive / negative / neutral.
  - Health‑check endpoint `/health` for monitoring and readiness.
  - `/metrics`  endpoint exposing Prometheus metrics from the FastAPI 
- **Model & Data**
  - Pretrained Hugging Face model (RoBERTa for Twitter sentiment), with support for fine-tuning.
  - Dataset: TweetEval sentiment dataset for training/validation.
  - Scripts for downloading dataset and fine-tuning the model.
  - In Docker, only lightweight API dependencies are installed; heavy model libraries are kept in the local dev       environment.
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

## Dockerized Service & Monitoring

## How This Repository Will Be Used

1. **Local development**
   - Clone the repo, create a virtual environment, install `requirements.txt`.
   - Run the FastAPI app locally (e.g. with `uvicorn`) and execute tests with `pytest`.
1.5 **Model Training**
   - Download the dataset: `python training/download_dataset.py`
   - Fine-tune the model: `python training/train_model.py` (requires GPU for speed)
   - The fine-tuned model will be saved in `model/fine_tuned_model` and used by the API.
2. **Dockerized API service**
   - Build and run the API container from the project root using the provided  `Dockerfile`  and  `docker-compose.yml `.
	 - Typical workflow:
	   - Start the API:
	     - `docker compose up -d --build`  
	   - Access the service:
	     - `http://localhost:8000/health`  – health‑check endpoint.
	     - `http://localhost:8000/docs`  – interactive API documentation.
	     - `http://localhost:8000/metrics`  – Prometheus metrics exposed by the FastAPI app.
	   - Stop the API:
	     - docker compose down 
3. **Monitoring stack (Prometheus+Grafana)**
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
     - only the minimal dependencies required to run FastAPI, expose Prometheus metrics and run integration tests are installed;
     - heavy model‑related libraries (e.g. full PyTorch + CUDA, large Transformers stacks) are not installed inside the API image and are instead used only in the local virtual environment.
     - This design still demonstrates a complete MLOps setup (API + CI/CD + monitoring) while keeping the Docker image small enough for Codespaces; full model execution and retraining can be done on a separate machine or image with more resources. 
  

