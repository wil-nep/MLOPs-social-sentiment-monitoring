# MLOps Social Sentiment Monitoring

**Public demo project** for monitoring company online reputation using social media sentiment analysis with MLOps practices.

## 🎯 Project Overview

Automated sentiment analysis API for social media monitoring, implementing:
- **Sentiment Analysis API** with FastAPI
- **CI/CD Pipeline** with GitHub Actions
- **Monitoring** with Prometheus and Grafana
- **Containerization** with Docker

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose
- GitHub account for CI/CD

## 📁 Repository Structure

```
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints and sentiment pipeline
│   └── __init__.py
├── tests/                  # Pytest test suite
│   └── test_api.py        # API and model tests
├── training/               # Reference scripts (fine-tuning demo)
│   ├── train_model.py
│   └── download_dataset.py
├── monitoring/             # Monitoring configurations
├── .github/workflows/      # CI/CD pipeline
│   └── ci-cd.yml
├── Dockerfile              # API container image
├── docker-compose.yml      # Full stack (API + monitoring)
├── prometheus.yml          # Prometheus configuration
├── requirements.txt        # Python dependencies
└── README.md
```

## ✨ Main Features

### 🤖 Sentiment Analysis API
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (pretrained)
- **Endpoints**:
  - `POST /predict` - Classify text sentiment (positive/negative/neutral)
  - `GET /health` - Health check
  - `GET /metrics` - Prometheus metrics
  - `GET /docs` - Interactive API documentation

### 🔄 CI/CD Pipeline
- Automated testing on every push/PR
- Docker image build and validation
- Health endpoint verification
- Optional Docker Hub deployment

### 📊 Monitoring Stack
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization dashboards
- **Metrics tracked**:
  - Request rate and latency
  - Error rates
  - Sentiment distribution over time

### 🧪 Testing
- pytest test suite
- API endpoint validation
- Model prediction sanity checks

## 🚀 Quick Start

### 1. Local Development

```bash
# Clone repository
git clone https://github.com/wil-nep/MLOPs-social-sentiment-monitoring.git
cd MLOPs-social-sentiment-monitoring

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API locally
uvicorn app.main:app --reload

# Run tests
pytest tests/ -v
```

Access API at:
- Health: http://localhost:8000/health
- Docs: http://localhost:8000/docs
- Metrics: http://localhost:8000/metrics

### 2. Docker Deployment

```bash
# Build and start all services (API + Prometheus + Grafana)
docker compose up -d --build

# Check status
docker compose ps

# View logs
docker compose logs -f api

# Stop services
docker compose down
```

Access services:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 3. API Usage Example

```bash
# Health check
curl http://localhost:8000/health

# Predict sentiment
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! I love it!"}'

# Response:
# {"label": "POSITIVE", "score": 0.95}
```

Or use the interactive docs at http://localhost:8000/docs

## 🔧 CI/CD Pipeline

The GitHub Actions pipeline (`github/workflows/ci-cd.yml`) automatically:

1. **Test Job** (on every push/PR):
   - Installs Python 3.11
   - Installs dependencies
   - Runs pytest test suite

2. **Build Job** (on push to main):
   - Builds Docker image
   - Starts container and validates health endpoint
   - Optionally pushes to Docker Hub (requires secrets)

### Required GitHub Secrets (Optional)

For Docker Hub deployment:
- `DOCKER_USERNAME` - Docker Hub username
- `DOCKER_PASSWORD` - Docker Hub password

## 📊 Monitoring Setup

### Prometheus

Configuration in `prometheus.yml`:
- Scrapes metrics from FastAPI app every 15s
- Available at http://localhost:9090

### Grafana

Default credentials: `admin/admin`

**Recommended dashboards:**
1. API Performance:
   - Request rate (requests/sec)
   - Response time percentiles (p50, p95, p99)
   - Error rate

2. Sentiment Analysis:
   - Sentiment distribution (positive/negative/neutral)
   - Prediction confidence scores
   - Processing time per request

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test
pytest tests/test_api.py::test_health_check -v
```

Test coverage includes:
- API endpoint responses
- Request/response schemas
- Model prediction validation
- Error handling

## 📚 Model Information

**Pretrained Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`

- **Architecture**: RoBERTa (Robustly Optimized BERT)
- **Training Data**: Twitter/social media posts
- **Classes**: POSITIVE, NEGATIVE, NEUTRAL
- **Performance**: ~70-75% accuracy on social media sentiment
- **Advantages**: 
  - Pre-trained on Twitter data (ideal for social media)
  - No fine-tuning required
  - Fast inference
  - Maintained by Cardiff NLP

## 🔄 Future Improvements

The `training/` directory contains reference scripts for:
- Dataset download (TweetEval)
- Fine-tuning the model
- Custom training pipelines

Note: Current implementation uses pretrained model as per project requirements.

## 🐛 Troubleshooting

### Docker Issues

```bash
# Clean Docker cache
docker system prune -a -f

# Check container logs
docker compose logs api

# Restart services
docker compose restart
```

### API Not Loading Model

The first request may take 10-15 seconds as the model downloads from Hugging Face.
Subsequent requests will be fast (model is cached).

### Space Issues (Codespaces)

```bash
# Check disk usage
df -h

# Clean Docker
docker system prune -a -f --volumes

# Remove old images
docker image prune -a -f
```

## 📝 Project Requirements

This project fulfills the following MLOps requirements:

✅ **Phase 1**: Sentiment analysis model (pretrained FastText/RoBERTa)  
✅ **Phase 2**: CI/CD pipeline with automated testing and deployment  
✅ **Phase 3**: Monitoring system with Prometheus and Grafana  
✅ **Documentation**: Comprehensive README and inline code comments  
✅ **Repository**: Public GitHub repository with clean structure  

## 👥 Contributing

This is a demo project for educational purposes. Contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Links

- **Model**: [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Dataset** (reference): [TweetEval](https://huggingface.co/datasets/tweet_eval)
- **Repository**: [GitHub](https://github.com/wil-nep/MLOPs-social-sentiment-monitoring)

---

**Built with**: FastAPI • Transformers • Docker • Prometheus • Grafana • GitHub Actions
