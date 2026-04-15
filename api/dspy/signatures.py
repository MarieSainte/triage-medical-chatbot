import dspy
import json
import re
import os
from pathlib import Path

# Chargement du JSON DSPY
_DSPY_JSON = Path(__file__).resolve().parent / "dspy_optimized_triage_dpo.json"

_DEFAULT_SYSTEM_PROMPT = (
    "Tu es un medecin urgentiste charge de trier des situations cliniques. "
    "Reponds UNIQUEMENT en JSON : "
    "{\"type\":\"final\",\"question\":null,\"urgence\":\"Haute|Moyenne|Faible\",\"analyse\":\"...\"} "
    "ou {\"type\":\"question\",\"question\":\"...\",\"urgence\":null,\"analyse\":null}"
)

def _load_optimized_prompt() -> str:
    if _DSPY_JSON.exists():
        try:
            with open(_DSPY_JSON, encoding="utf-8") as f:
                prog = json.load(f)
            return prog.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        except Exception:
            pass
    return _DEFAULT_SYSTEM_PROMPT

OPTIMIZED_SYSTEM_PROMPT = _load_optimized_prompt()


class TriageSignature(dspy.Signature):
    """Placeholder — remplacé dynamiquement par le prompt DSPy optimisé."""
    symptomes = dspy.InputField(desc="Symptômes ou question du patient")
    reponse   = dspy.OutputField(
        desc=(
            'JSON strict : {"type":"final","question":null,"urgence":"Haute|Moyenne|Faible","analyse":"..."} '
            'ou {"type":"question","question":"...","urgence":null,"analyse":null}'
        )
    )

# Injection du prompt optimisé comme instructions de la signature
TriageSignature.__doc__ = OPTIMIZED_SYSTEM_PROMPT


def _extract_json(raw: str) -> dict | None:
    """Extrait le premier bloc JSON valide depuis la réponse brute du modèle."""
    raw = raw.strip()
    # Nettoyer les balises markdown
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
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
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(TriageSignature)

    def forward(self, symptomes: str) -> dict:
        result = self.predictor(symptomes=symptomes)
        raw_text = result.reponse

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

        # Demander plus d'informations
        return {
            "status": "ASSISTANT",
            "question": "Pouvez-vous me donner plus de détails sur vos symptômes ?",
        }
