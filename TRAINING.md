# Training and Fine-Tuning Guide

## Overview

Questo progetto implementa un sistema completo di retraining automatico per il modello di analisi del sentimento.

## Scripts

### 1. download_dataset.py
Scarica e prepara i dataset per l'addestramento.

**Uso:**
```bash
# Scaricare e preparare i dataset di sentiment
python training/download_dataset.py

# Verificare e cachare il modello pretrained
python training/download_dataset.py model
```

**Output:**
- Dataset salvati in `data/`
- Informazioni dataset in `data/dataset_info.txt`
- Modello pretrained in cache HuggingFace

### 2. train_model.py
Fine-tuning del modello con validazione periodica.

**Features:**
- ✅ Fine-tuning su dataset sentiment reale
- ✅ Early stopping basato su F1 score
- ✅ Validazione automatica
- ✅ Salvataggio checkpoint e best model
- ✅ Metriche dettagliate (accuracy, F1, precision, recall)
- ✅ Support per ripresa training da checkpoint

**Uso:**
```bash
# Allenare un nuovo modello
python training/train_model.py

# Riprendere l'allenamento da ultimo checkpoint
python training/train_model.py resume

# Validare un modello specifico
python training/train_model.py validate /path/to/model

# Validare il modello più recente
python training/train_model.py validate
```

**Output:**
- Modelli salvati in `models/model-YYYYMMDD-HHMMSS/`
- Metriche in `metrics/training_metrics.json`
- Tensorboard logs in `models/checkpoint/`

## Training Configuration

```python
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 2
```

Modificare in `train_model.py` per adattare ai vostri bisogni.

## Workflow Automatico

### Per integrazione CI/CD:

```bash
# 1. Download dataset
python training/download_dataset.py

# 2. Allenare il modello
python training/train_model.py

# 3. Validare il modello
python training/train_model.py validate

# 4. (Opzionale) Aggiornare l'API con il nuovo modello
```

## Metriche di Training

Ogni training genera un file `metrics/training_metrics.json`:

```json
{
  "timestamp": "2026-01-06T10:30:45.123456",
  "model_path": "models/model-20260106-103045",
  "training_metrics": {
    "loss": 0.45,
    "learning_rate": 2e-5,
    ...
  },
  "evaluation_metrics": {
    "accuracy": 0.92,
    "f1": 0.91,
    "precision": 0.93,
    "recall": 0.89
  },
  "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

## Dataset

Il sistema supporta automaticamente:
1. **Tweet Eval Dataset** (principale): 9,100 tweet precaricati
2. **Twitter Sentiment 2015** (fallback): Dataset alternativo

Con fallback su dati sintetici se i dataset remoti non sono disponibili.

## Integrazione Docker

Per addestrare dentro Docker:

```bash
docker-compose build
docker run --rm -v huggingface-cache:/root/.cache/huggingface \
  mlops-social-sentiment-monitoring python training/train_model.py
```

## Validazione Periodica

Implementare una validazione periodica nel vostro orchestrator (Kubernetes, Airflow, etc.):

```bash
# Ogni settimana
0 2 * * 0 python training/train_model.py validate

# Se accuracy < 0.85, triggerare nuovo training
```

## Note Importanti

- ⚠️ Il primo run scaricherà il modello (~600MB) - ci vuole tempo
- 💾 I volumi Docker con cache HuggingFace evitano download ripetuti
- 🔄 Early stopping ferma il training se F1 non migliora per N epoch
- 📊 Tutte le metriche vengono salvate per monitoring e logging
