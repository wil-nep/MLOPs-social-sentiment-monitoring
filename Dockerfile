FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "tokenizers>=0.19,<0.20" regex safetensors && \
    pip install --no-cache-dir --no-deps transformers==4.41.2 accelerate==0.30.1 && \
    pip install --no-cache-dir fastapi==0.115.0 uvicorn==0.30.1 pydantic==2.8.2 prometheus-fastapi-instrumentator==6.1.0 prometheus_client==0.20.0 psutil==5.9.8 numpy==1.26.4 scikit-learn==1.5.1 pandas==2.1.4 datasets==2.14.6

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
