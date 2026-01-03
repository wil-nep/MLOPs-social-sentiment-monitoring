# MLOps Social Sentiment Monitoring

Automated sentiment analysis API for social media monitoring with MLOps practices.

## 🎯 Features

- **Sentiment Analysis API** (FastAPI) with pretrained RoBERTa model
- **CI/CD Pipeline** (GitHub Actions) - automated testing & deployment
- **Monitoring Stack** (Prometheus + Grafana)
- **Docker** containerization

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
├── monitoring/       # Prometheus + Grafana configs
├── .github/          # CI/CD pipeline
└── docker-compose.yml
```

## 🔧 CI/CD Pipeline

GitHub Actions automatically:
- Runs tests on every push/PR
- Builds and validates Docker image
- Optionally pushes to Docker Hub (requires `DOCKER_USERNAME` & `DOCKER_PASSWORD` secrets)

## 📊 Monitoring

**Prometheus** scrapes `/metrics` endpoint every 15s  
**Grafana** dashboards track:
- Request rate & latency
- Sentiment distribution
- Error rates

## 🧪 Testing

```bash
pytest tests/ -v                          # Run all tests
pytest tests/ --cov=app --cov-report=html # With coverage
```

## 📚 Model

**Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`  
**Type**: Pretrained RoBERTa for Twitter sentiment  
**Classes**: POSITIVE, NEGATIVE, NEUTRAL  
**Performance**: ~70-75% accuracy on social media text

## 🐛 Troubleshooting

```bash
# Docker cleanup
docker system prune -a -f

# View logs
docker compose logs api

# First API call may take 10-15s (model download)
```

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
