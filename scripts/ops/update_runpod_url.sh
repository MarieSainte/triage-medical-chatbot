#!/bin/bash
# Met a jour VLLM_API_URL sur la VM OVH avec l'URL du pod RunPod courant,
# puis recree le conteneur api pour qu'il prenne la nouvelle URL.
# A lancer en local (git-bash) apres avoir demarre vLLM sur le pod.
#
# Usage :
#   ./update_runpod_url.sh <POD_ID>
#   POD_ID = partie avant le "-" de l'URL SSH RunPod
#            (ex: et4hiz76s8jjh2-64410b77@ssh.runpod.io -> et4hiz76s8jjh2)
#   On accepte aussi l'URL SSH complete, l'ID est extrait automatiquement.

set -euo pipefail

# --- Config VM OVH (lue depuis le .env du repo, secrets hors du script) ---
_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_ENV" ] && { set -a; . "$_ENV"; set +a; }

OVH_HOST="${OVH_HOST:?OVH_HOST manquant (a definir dans .env)}"
OVH_USER="${OVH_USER:?OVH_USER manquant (a definir dans .env)}"
OVH_KEY="${OVH_KEY:?OVH_KEY manquant (a definir dans .env)}"
OVH_DEPLOY_PATH="${OVH_DEPLOY_PATH:?OVH_DEPLOY_PATH manquant (a definir dans .env)}"

# --- Pod ID + URL proxy ---
RAW="${1:?Usage: $0 <POD_ID | url_ssh_runpod>}"
POD_ID="${RAW%%-*}"                                # garde tout avant le premier "-"
VLLM_URL="https://${POD_ID}-8000.proxy.runpod.net/v1"

echo "Pod ID   : $POD_ID"
echo "VLLM URL : $VLLM_URL"

# --- Garde : le pod sert-il vraiment vLLM ? ---
echo "Verif que vLLM repond sur le pod..."
resp="$(curl -sS --max-time 25 "$VLLM_URL/models" 2>/dev/null || true)"
if ! echo "$resp" | grep -q '"object"\|medical'; then
  echo "ERREUR : le pod ne sert pas encore vLLM sur 8000."
  echo "  Demarre-le d'abord sur le pod (supervisorctl start vllm), puis relance."
  exit 1
fi
echo "  vLLM OK."

# --- Maj .env + recreation de l'api sur OVH ---
ssh -i "$OVH_KEY" "${OVH_USER}@${OVH_HOST}" bash -s <<EOF
set -euo pipefail
cd "$OVH_DEPLOY_PATH"
if grep -q '^VLLM_API_URL=' .env; then
  sed -i "s#^VLLM_API_URL=.*#VLLM_API_URL=$VLLM_URL#" .env
else
  echo "VLLM_API_URL=$VLLM_URL" >> .env
fi
echo "-> .env mis a jour :"; grep '^VLLM_API_URL=' .env
docker compose -f docker-compose.prod.yml up -d --force-recreate api
EOF

echo "Termine. L'API OVH pointe maintenant vers le pod $POD_ID."
echo "Test : curl -s http://$OVH_HOST:8080/health"
