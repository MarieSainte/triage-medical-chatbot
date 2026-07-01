#!/bin/bash
# Script de setup complet pour le pod RunPod (services natifs via supervisord)
# Usage: bash setup_pod_native.sh <HF_TOKEN> <GITHUB_REPO_URL>
set -e

HF_TOKEN="${1:-}"
REPO_URL="${2:-https://github.com/MarieSainte/medical-chatbot.git}"
WORKSPACE="/workspace"
VENV="$WORKSPACE/venv"

echo "=== [1/6] Clonage du repo ==="
if [ -d "$WORKSPACE/medical-chatbot/.git" ]; then
    cd "$WORKSPACE/medical-chatbot" && git pull
else
    git clone "$REPO_URL" "$WORKSPACE/medical-chatbot"
fi

echo "=== [2/6] Telechargement du modele depuis HuggingFace ==="
if [ ! -f "$WORKSPACE/production_model/config.json" ]; then
    $VENV/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='huggingjojo/medical-chatbot-model',
    local_dir='$WORKSPACE/production_model',
    token='$HF_TOKEN'
)
print('Modele telecharge.')
"
else
    echo "Modele deja present, skip."
fi

echo "=== [3/6] Installation des deps API ==="
cd "$WORKSPACE/medical-chatbot"
$VENV/bin/pip install -q --no-cache-dir -r api/requirements.txt
$VENV/bin/pip install -q --no-cache-dir streamlit

echo "=== [4/6] Configuration PostgreSQL ==="
service postgresql start || true
sleep 3
sudo -u postgres psql -c "CREATE USER chatbot WITH PASSWORD 'chatbot123';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE chatbot_db OWNER chatbot;" 2>/dev/null || true

echo "=== [5/6] Configuration supervisord ==="
mkdir -p /var/log/supervisor
mkdir -p /etc/supervisor/conf.d

cat > /etc/supervisor/conf.d/medical-chatbot.conf << 'SUPERVISOREOF'
[supervisord]
nodaemon=false
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid
childlogdir=/var/log/supervisor

[unix_http_server]
file=/var/run/supervisor.sock

[supervisorctl]
serverurl=unix:///var/run/supervisor.sock

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[program:vllm]
command=/workspace/venv/bin/python -m vllm.entrypoints.openai.api_server
    --model /workspace/production_model
    --served-model-name medical_chatbot
    --host 0.0.0.0
    --port 8000
    --dtype bfloat16
    --max-model-len 4096
    --gpu-memory-utilization 0.7
directory=/workspace
environment=HOME="/root",XDG_CACHE_HOME="/tmp/.cache"
autostart=true
autorestart=true
startsecs=60
startretries=3
stdout_logfile=/var/log/supervisor/vllm.log
stderr_logfile=/var/log/supervisor/vllm_err.log

[program:fastapi]
command=/workspace/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8080 --workers 1
directory=/workspace/medical-chatbot
environment=
    DATABASE_URL="postgresql://chatbot:chatbot123@localhost/chatbot_db",
    VLLM_API_URL="http://localhost:8000/v1",
    HOME="/root"
autostart=true
autorestart=true
startsecs=10
startretries=5
stdout_logfile=/var/log/supervisor/fastapi.log
stderr_logfile=/var/log/supervisor/fastapi_err.log

[program:streamlit]
command=/workspace/venv/bin/streamlit run interface/app.py
    --server.port 8501
    --server.address 0.0.0.0
    --server.headless true
directory=/workspace/medical-chatbot
environment=API_URL="http://localhost:8080",HOME="/root"
autostart=true
autorestart=true
startsecs=10
startretries=3
stdout_logfile=/var/log/supervisor/streamlit.log
stderr_logfile=/var/log/supervisor/streamlit_err.log

SUPERVISOREOF

echo "=== [6/6] Installation et config Prometheus + Grafana ==="

# Prometheus
if [ ! -f /usr/local/bin/prometheus ]; then
    PROM_VERSION="2.51.2"
    cd /tmp
    wget -q "https://github.com/prometheus/prometheus/releases/download/v${PROM_VERSION}/prometheus-${PROM_VERSION}.linux-amd64.tar.gz"
    tar -xzf "prometheus-${PROM_VERSION}.linux-amd64.tar.gz"
    cp "prometheus-${PROM_VERSION}.linux-amd64/prometheus" /usr/local/bin/
    mkdir -p /workspace/prometheus_data
    echo "Prometheus installe"
fi

cat >> /etc/supervisor/conf.d/medical-chatbot.conf << 'PROMEOF'

[program:prometheus]
command=/usr/local/bin/prometheus
    --config.file=/workspace/medical-chatbot/monitoring/prometheus.yml
    --storage.tsdb.path=/workspace/prometheus_data
    --web.listen-address=0.0.0.0:9090
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/prometheus.log
stderr_logfile=/var/log/supervisor/prometheus_err.log

PROMEOF

# Grafana
if [ ! -f /usr/sbin/grafana-server ]; then
    apt-get install -y -qq apt-transport-https software-properties-common 2>/dev/null || true
    wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor > /etc/apt/keyrings/grafana.gpg 2>/dev/null || true
    echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" > /etc/apt/sources.list.d/grafana.list
    apt-get update -qq 2>/dev/null || true
    apt-get install -y -qq grafana 2>/dev/null || true
    echo "Grafana installe"
fi

# Provisioning Grafana
mkdir -p /etc/grafana/provisioning/datasources /etc/grafana/provisioning/dashboards
cp /workspace/medical-chatbot/monitoring/grafana/provisioning/datasources/prometheus.yml /etc/grafana/provisioning/datasources/
cp /workspace/medical-chatbot/monitoring/grafana/provisioning/dashboards/dashboards.yml /etc/grafana/provisioning/dashboards/

cat >> /etc/supervisor/conf.d/medical-chatbot.conf << 'GRAFANAEOF'

[program:grafana]
command=/usr/sbin/grafana-server
    --config=/etc/grafana/grafana.ini
    --homepath=/usr/share/grafana
environment=
    GF_SERVER_HTTP_PORT="3000",
    GF_SECURITY_ADMIN_PASSWORD="admin123",
    GF_AUTH_ANONYMOUS_ENABLED="true",
    GF_AUTH_ANONYMOUS_ORG_ROLE="Viewer",
    HOME="/root"
autostart=true
autorestart=true
stdout_logfile=/var/log/supervisor/grafana.log
stderr_logfile=/var/log/supervisor/grafana_err.log

GRAFANAEOF

echo ""
echo "=== Demarrage de supervisord ==="
supervisord -c /etc/supervisor/conf.d/medical-chatbot.conf

echo ""
echo "========================================"
echo "Stack lancee ! Ports RunPod a exposer :"
echo "  8080  -> FastAPI (triage API)"
echo "  8501  -> Streamlit (interface)"
echo "  3000  -> Grafana (monitoring)"
echo "  9090  -> Prometheus (metrics raw)"
echo ""
echo "Vérifier: supervisorctl -c /etc/supervisor/conf.d/medical-chatbot.conf status"
echo "Logs vLLM: tail -f /var/log/supervisor/vllm_err.log"
echo "========================================"
