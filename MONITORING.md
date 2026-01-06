# Docker Compose Monitoring - Consolidated Setup

## File Structure After Consolidation

Il `docker-compose.yml` principale ora include:
- API FastAPI con Prometheus instrumentation
- Prometheus per metriche
- Grafana per visualizzazione
- HuggingFace cache volume per il modello

Il file `monitoring/docker-compose.monitoring.yml` è stato corretto e rimane come riferimento standalone.

## Usage

### Option 1: Avvio combinato (consigliato)
```bash
docker-compose up -d
```

Questo avvia:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Health check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

### Option 2: Monitoring standalone
```bash
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

## Prometheus Configuration

File: `monitoring/prometheus.yml`

Scrape targets:
- api:8000/metrics (metriche applicazione)

Retention: 15 giorni

## Grafana Setup

1. Login: http://localhost:3000
   - Username: admin
   - Password: admin123

2. Add Prometheus data source:
   - Name: Prometheus
   - URL: http://prometheus:9090

3. Import dashboards per:
   - Request latency
   - Sentiment prediction distribution
   - Model inference performance

## Volumes

```yaml
volumes:
  prometheus_data:  # Metriche Prometheus
  grafana_data:     # Configuration Grafana
  huggingface-cache: # Modello pretrained (~600MB)
```

Volume huggingface-cache è persistente - evita download ripetuti del modello.

## Health Checks

```bash
# API status
curl http://localhost:8000/health

# Prometheus status
curl http://localhost:9090/-/healthy

# Scrape targets
curl http://localhost:9090/api/v1/targets
```

## Logs

```bash
# Logs API
docker logs <api-container-id>

# Logs Prometheus
docker logs prometheus

# Logs Grafana
docker logs grafana
```

## Cleanup

```bash
docker-compose down
docker volume rm mlops-social-sentiment-monitoring_prometheus_data mlops-social-sentiment-monitoring_grafana_data
```
