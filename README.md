# MLOps Social Sentiment Monitoring

Automated sentiment analysis API for social media monitoring with MLOps practices.

## 🎯 Features

- **Sentiment Analysis API** (FastAPI) with pretrained RoBERTa model
- **Automatic Model Retraining** - fine-tuning, validation, and metrics tracking
- **CI/CD Pipeline** (GitHub Actions) - automated testing & deployment
- **Monitoring Stack** (Prometheus + Grafana)
- **Docker** containerization with HuggingFace cache persistence

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose

## 🚀 Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/wil-nep/MLOPs-social-sentiment-monitoring.git
cd MLOPs-social-sentiment-monitoring
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload  # API at http://localhost:8000
pytest tests/ -v                # Run tests
```

### Docker Deployment

```bash
docker compose up -d --build
```

**Access:**
- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

## 📁 Structure

```
├── app/              # FastAPI application
├── tests/            # pytest suite
├── training/         # Model training & fine-tuning
├── monitoring/       # Prometheus + Grafana configs
├── .github/          # CI/CD pipeline
├── TRAINING.md       # Training guide
├── MONITORING.md     # Monitoring setup
└── docker-compose.yml
```

## 🔧 Training & Fine-Tuning

Fully implemented model retraining system:

```bash
# Download datasets and cache pretrained model
python training/download_dataset.py

# Fine-tune model on sentiment data
python training/train_model.py

# Validate model performance
python training/train_model.py validate

# Resume training from checkpoint
python training/train_model.py resume
```

**Features:**
- ✅ Automatic dataset loading (Tweet Eval + fallback to synthetic data)
- ✅ Fine-tuning with early stopping
- ✅ Comprehensive metrics (accuracy, F1, precision, recall)
- ✅ Checkpoint management
- ✅ Metrics logging (`metrics/training_metrics.json`)

See [TRAINING.md](TRAINING.md) for detailed guide.

## 📊 Monitoring

**Prometheus** scrapes `/metrics` endpoint every 15s  
**Grafana** dashboards track:
- Request rate & latency
- Sentiment distribution
- Error rates
- Model performance metrics

Setup includes both combined (`docker-compose.yml`) and standalone (`monitoring/docker-compose.monitoring.yml`) configurations.

See [MONITORING.md](MONITORING.md) for detailed setup.

## 🔄 CI/CD Pipeline

GitHub Actions automatically:
- Runs tests on every push/PR
- Builds and validates Docker image
- Optionally pushes to Docker Hub (requires `DOCKER_USERNAME` & `DOCKER_PASSWORD` secrets)

## 🧪 Testing

```bash
pytest tests/ -v                          # Run all tests
pytest tests/ --cov=app --cov-report=html # With coverage
```

## 📚 Model

**Base Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`  
**Type**: Pretrained RoBERTa for Twitter sentiment  
**Classes**: POSITIVE, NEGATIVE, NEUTRAL  
**Fine-tuning**: Automatic on real sentiment datasets  

## 🐛 Troubleshooting

```bash
# Docker cleanup
docker system prune -a -f

# View logs
docker compose logs api

# First API call may take 10-15s (model download)
# HuggingFace cache volume prevents re-downloads

# Check training status
tail -f metrics/training_metrics.json
```

## 📝 Configuration

### Training Config (training/train_model.py)
- `BATCH_SIZE`: 16
- `EPOCHS`: 3
- `LEARNING_RATE`: 2e-5
- `EARLY_STOPPING_PATIENCE`: 2

### Docker Volumes
- `huggingface-cache`: Persistent model cache (~600MB)
- `prometheus_data`: Prometheus metrics
- `grafana_data`: Grafana dashboards & configs

## 📝 Requirements Met

✅ Sentiment analysis (pretrained model)  
✅ CI/CD pipeline  
✅ Monitoring (Prometheus + Grafana)  
✅ Automated testing  
✅ Docker containerization  
✅ Documentation

## 🔗 Links

- [Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [Repository](https://github.com/wil-nep/MLOPs-social-sentiment-monitoring)

---

**Stack**: FastAPI • Transformers • Docker • Prometheus • Grafana • GitHub Actions
