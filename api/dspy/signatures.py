import dspy
import json
import re
import os
from pathlib import Path

_DSPY_DIR = Path(__file__).resolve().parent
_DSPY_PROFILE = os.getenv("DSPY_PROFILE", "sft").lower()
_DSPY_PROMPT_FILE = os.getenv("DSPY_PROMPT_FILE")

# Par defaut, on utilise le prompt DSPy SFT valide sur la gate CI. Un override
# explicite reste possible par variable d'environnement pour cibler le DPO ou
# un autre fichier JSON sans reedition du code.
if _DSPY_PROMPT_FILE:
    _DSPY_JSON = Path(_DSPY_PROMPT_FILE)
    if not _DSPY_JSON.is_absolute():
        _DSPY_JSON = _DSPY_DIR / _DSPY_JSON
else:
    _DSPY_JSON = _DSPY_DIR / f"dspy_optimized_triage_{_DSPY_PROFILE}.json"

_DEFAULT_SYSTEM_PROMPT = (
    "Tu es un medecin urgentiste charge de trier des situations cliniques.\n"
    "Reponds UNIQUEMENT en JSON strict, sans texte avant ni apres :\n"
    "{\"type\":\"final\",\"question\":null,\"urgence\":\"Haute|Moyenne|Faible\",\"analyse\":\"...\"}\n"
    "ou\n"
    "{\"type\":\"question\",\"question\":\"...\",\"urgence\":null,\"analyse\":null}"
)


def _load_config() -> dict:
    if _DSPY_JSON.exists():
        try:
            with open(_DSPY_JSON, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_CONFIG = _load_config()
OPTIMIZED_SYSTEM_PROMPT = _CONFIG.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
_DEMOS = _CONFIG.get("demos", [])


def _build_messages(symptomes: str) -> list:
    """
    Construit les messages pour l'appel LM.
    Le system prompt contient déjà les exemples en texte — ne pas réinjecter
    les démos comme turns conversation (le modèle SFT les a vus à l'entraînement).
    """
    return [
        {"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT},
        {"role": "user", "content": symptomes},
    ]


def _extract_json(raw: str) -> dict | None:
    """Extrait le premier bloc JSON valide depuis la réponse brute du modèle."""
    raw = raw.strip()
    # Nettoyer les balises markdown
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    # Nettoyer les balises DSPy Predict si présentes
    raw = re.sub(r"\[\[\s*##\s*\w+\s*##\s*\]\]", "", raw).strip()
    # Trouver le premier JSON valide
    brace = raw.find("{")
    if brace == -1:
        return None
    for end in range(len(raw), brace, -1):
        try:
            return json.loads(raw[brace:end])
        except Exception:
            continue
    return None


class TriageModule(dspy.Module):
    """
    Appelle directement dspy.LM avec le format chat (messages),
    en bypassant dspy.Predict pour éviter le wrapping de tokens DSPy
    incompatible avec le modèle fine-tuné.
    max_tokens fixé à 256 pour éviter la troncature.
    """

    def __init__(self):
        super().__init__()

    def forward(self, symptomes: str = None, messages: list = None) -> dict:
        lm = dspy.settings.lm

        if messages:
            # Historique ChatML complet fourni par l'interface
            # On prépend system prompt + démos few-shot si pas déjà de system message
            if not messages or messages[0].get("role") != "system":
                full_messages = [{"role": "system", "content": OPTIMIZED_SYSTEM_PROMPT}]
                full_messages.extend(messages)
            else:
                full_messages = messages
        else:
            # Fallback : appel direct avec un seul message utilisateur
            full_messages = _build_messages(symptomes or "")

        # Appel direct au LM — retourne une liste de strings
        raw_responses = lm(
            messages=full_messages,
            max_tokens=256,
            temperature=0.1,
        )

        # dspy.LM retourne une liste, on prend le premier élément
        raw_text = raw_responses[0] if isinstance(raw_responses, list) else raw_responses

        data = _extract_json(raw_text)

        if data and data.get("type") == "final":
            urgence = data.get("urgence")
            analyse = data.get("analyse", "")
            if urgence in {"Haute", "Moyenne", "Faible"} and analyse:
                return {
                    "status": "ANALYSE",
                    "data": {"urgence": urgence, "analyse": analyse},
                }

        if data and data.get("type") == "question":
            question = data.get("question", "")
            if question:
                return {"status": "ASSISTANT", "question": question}

        # Fallback si JSON non parseable
        return {
            "status": "ASSISTANT",
            "question": "Pouvez-vous me donner plus de détails sur vos symptômes ?",
        }
