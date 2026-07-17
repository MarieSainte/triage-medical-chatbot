#!/bin/bash
# Setup RunPod : vLLM SEUL (l'API/interface/monitoring tournent sur OVH).
# vLLM sert huggingjojo/medical-chatbot-model (repo HF prive) et le tire depuis
# HF au demarrage. Cache dans /workspace -> persiste ; sur restart, HF re-verifie
# la revision `main` -> le nouveau DPO est pris automatiquement (pas de skip aveugle).
#
# Prerequis :
#   - volume persistant sur /workspace (Network Volume conseille pour survivre a
#     la duplication du pod ; sinon re-download a chaque nouveau pod)
#   - HF_TOKEN dans l'env (ou dans /workspace/.env)
#
# Usage sur le pod :
#   export HF_TOKEN=hf_xxx        # ou deposer /workspace/.env avec HF_TOKEN=...
#   bash setup_pod_native.sh
#
# Ensuite, exposer le port 8000 (HTTP) sur RunPod, puis en local :
#   bash scripts/ops/update_runpod_url.sh <POD_ID>

set -euo pipefail

WORKSPACE="/workspace"
MODEL_REPO="${MODEL_REPO:-huggingjojo/medical-chatbot-model}"
VENV="$WORKSPACE/venv"
CONF="/etc/supervisor/conf.d/vllm.conf"

# HF_TOKEN depuis l'env, ou depuis un .env pose sur le pod.
# On cherche le .env a plusieurs endroits et on nettoie les CRLF Windows (\r)
# avant de le sourcer, sinon HF_TOKEN vaudrait "hf_xxx\r" (token invalide) ou
# le sourcing casserait. Le fichier n'est pas modifie : on source une copie nettoyee.
_load_env() {
  local f="$1"
  [ -f "$f" ] || return 1
  set -a
  # shellcheck disable=SC1090
  . <(sed 's/\r$//' "$f")
  set +a
  echo "  .env charge depuis : $f"
}

# Emplacements candidats : /workspace/.env (pose sur le pod), puis le .env du repo
# s'il a ete copie a cote du script.
_REPO_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)/.env"
_load_env "$WORKSPACE/.env" || _load_env "$_REPO_ENV" || true

# Nettoie un eventuel \r sur un HF_TOKEN venu de l'environnement (export sous Windows).
HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN="${HF_TOKEN%$'\r'}"

: "${HF_TOKEN:?HF_TOKEN requis (repo HF prive) — export HF_TOKEN=... ou /workspace/.env}"

export HF_HOME="$WORKSPACE/hf_cache"          # cache persistant
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" /var/log/supervisor /etc/supervisor/conf.d

echo "=== [1/3] venv + vLLM ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q vllm huggingface_hub hf_transfer

echo "=== [2/3] telechargement/maj du modele (revision main la plus recente) ==="
HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" "$VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_REPO")   # re-verifie main, ne re-telecharge que ce qui a change
print("Modele en cache :", p)
PY

echo "=== [3/3] supervisord (vLLM seul) ==="
cat > "$CONF" <<SUP
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
command=$VENV/bin/python -m vllm.entrypoints.openai.api_server
    --model $MODEL_REPO
    --served-model-name medical_chatbot
    --host 0.0.0.0
    --port 8000
    --dtype bfloat16
    --max-model-len 2048
    --gpu-memory-utilization 0.9
environment=HF_HOME="$HF_HOME",HF_TOKEN="$HF_TOKEN",HF_HUB_ENABLE_HF_TRANSFER="1"
directory=$WORKSPACE
autostart=true
autorestart=true
startsecs=60
startretries=3
stdout_logfile=/var/log/supervisor/vllm.log
stderr_logfile=/var/log/supervisor/vllm_err.log
SUP

# (re)demarrage : restart si supervisord tourne deja, sinon le lance
if supervisorctl -c "$CONF" status >/dev/null 2>&1; then
  supervisorctl -c "$CONF" reread
  supervisorctl -c "$CONF" update
  supervisorctl -c "$CONF" restart vllm
else
  supervisord -c "$CONF"
fi

echo ""
echo "vLLM demarre. Exposer le port 8000 (HTTP) sur RunPod."
echo "Verifier : curl -s http://localhost:8000/v1/models   (attendre 'medical_chatbot')"
echo "Logs      : tail -f /var/log/supervisor/vllm_err.log"
