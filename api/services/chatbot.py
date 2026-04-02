import os
import logging
from sqlalchemy.orm import Session
from openai import APIConnectionError, APITimeoutError
from api.database import models
import dspy
from api.dspy.signatures import TriageModule

# Configuration du logger pour voir les erreurs dans les logs Docker/GCP
logger = logging.getLogger(__name__)

VLLM_API_URL = os.getenv("VLLM_API_URL")

vllm_endpoint = dspy.HFClientVLLM(
    model="medical_lora", 
    port=8000, 
    url=os.getenv("VLLM_API_URL")
)
dspy.settings.configure(lm=vllm_endpoint)

triage_app = TriageModule()

json_path = os.path.join(os.path.dirname(__file__), "..", "dspy", "optimized_triage.json")
if os.path.exists(json_path):
    triage_app.load(json_path)
    logger.info("Cerveau DSPy chargé avec succès depuis le JSON.")
else:
    logger.warning("Fichier optimized_triage.json introuvable, utilisation du mode par défaut.")

def generate_triage(symptomes: str) -> str:
    """
    Appelle le modèle d'IA avec gestion d'erreurs robuste.
    """
    try:
        prediction = triage_module(symptomes=symptomes)

        reponse_formatee = (
            f"### {prediction.niveau_urgence}\n"
            f"**Pourquoi** : {prediction.justification}\n"
            f"**Action** : {prediction.action}"
        )
        return reponse_formatee

    except APITimeoutError:
        logger.error("Timeout: vLLM a mis trop de temps à répondre.")
        return "Désolé, le service de diagnostic est surchargé. Veuillez réessayer dans un instant."
    
    except APIConnectionError:
        logger.error("Connexion impossible au serveur vLLM. Est-il lancé ?")
        return "Désolé, un problème technique empêche la réponse immédiate. Nos équipes sont prévenues."
    
    except Exception as e:
        logger.error(f"Erreur inconnue vLLM: {str(e)}")
        return "Désolé, une erreur technique est survenue lors de l'analyse de vos symptômes."

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