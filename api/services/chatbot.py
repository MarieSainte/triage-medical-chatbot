import os
import logging
import time
import json
from sqlalchemy.orm import Session
from openai import APIConnectionError, APITimeoutError
from prometheus_client import Counter, Histogram
from database import models
import dspy
import importlib.util

logger = logging.getLogger("medical-chatbot-api.triage")

TRIAGE_COUNTER = Counter(
    "triage_requests_total",
    "Nombre total de triages",
    ["status", "urgence"],
)
TRIAGE_LATENCY = Histogram(
    "triage_latency_seconds",
    "Latence des appels au modele IA",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

spec = importlib.util.spec_from_file_location("local_signatures", os.path.join(os.path.dirname(__file__), "..", "dspy", "signatures.py"))
local_signatures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_signatures)
TriageModule = local_signatures.TriageModule


VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")

vllm_endpoint = dspy.LM(
    model="openai/medical_lora",
    api_base=VLLM_API_URL,
    api_key="EMPTY"
)
dspy.settings.configure(lm=vllm_endpoint)

triage_app = TriageModule()

def generate_triage(symptomes: str) -> dict:
    """
    Appelle le modèle d'IA avec gestion d'erreurs robuste et mesure de la latence.
    """
    start_time = time.time()
    try:
        prediction = triage_app(symptomes=symptomes)
        latency = round(time.time() - start_time, 2)
        prediction["latency"] = latency

        urgence = (prediction.get("data") or {}).get("urgence") if prediction.get("status") == "ANALYSE" else "none"
        TRIAGE_COUNTER.labels(status=prediction.get("status", "UNKNOWN"), urgence=urgence or "none").inc()
        TRIAGE_LATENCY.observe(latency)
        logger.info(
            "triage_completed",
            extra={
                "event": "triage_completed",
                "status": prediction.get("status"),
                "urgence": urgence,
            },
        )
        return prediction

    except APITimeoutError:
        latency = round(time.time() - start_time, 2)
        TRIAGE_COUNTER.labels(status="ERROR", urgence="none").inc()
        TRIAGE_LATENCY.observe(latency)
        logger.error("triage_timeout", extra={"event": "triage_error", "error_type": "timeout"})
        return {"status": "ERROR", "message": "Service surchargé.", "latency": latency}

    except APIConnectionError:
        latency = round(time.time() - start_time, 2)
        TRIAGE_COUNTER.labels(status="ERROR", urgence="none").inc()
        TRIAGE_LATENCY.observe(latency)
        logger.error("triage_connection_error", extra={"event": "triage_error", "error_type": "connection"})
        return {"status": "ERROR", "message": "Connexion impossible à l'IA.", "latency": latency}

    except Exception as e:
        latency = round(time.time() - start_time, 2)
        TRIAGE_COUNTER.labels(status="ERROR", urgence="none").inc()
        TRIAGE_LATENCY.observe(latency)
        logger.error("triage_unknown_error", extra={"event": "triage_error", "error_type": "unknown", "detail": str(e)})
        return {"status": "ERROR", "message": "Erreur technique.", "latency": latency}

def log_triage(db: Session, question: str, answer: str):
    """
    Sauvegarde l'échange, même si c'est un message d'erreur.
    """
    try:
        db_history = models.History(question=question, answer=answer)
        db.add(db_history)
        db.commit()
        db.refresh(db_history)
    except Exception as e:
        logger.error(f"Erreur lors de l'écriture en BDD: {str(e)}")
        db.rollback()

def get_logs(db: Session):
    """
    Récupère l'historique complet des échanges.
    """
    try:
        return db.query(models.History).order_by(models.History.created_at.desc()).all()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {str(e)}")
        return []
