import os
import json
import time
import random
from datetime import date
from pathlib import Path

from mistralai.client import Mistral
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DATASET_PATH = BASE_DIR / "data" / "data_versioned" / "data_sft_v1.0.0_reviewed_mixed.jsonl"
OUTPUT_DATASET_PATH = BASE_DIR / "data" / "data_versioned" / "data_sft_v1.0.0_reviewed_final.jsonl"

N_NEW_CASES = 400
BATCH_SIZE = 2

MODEL_NAME = "mistral-small-latest"
TEMPERATURE = 0.8
MAX_RETRIES = 5
SLEEP_BETWEEN_CALLS = 0.5
SHUFFLE_SEED = 42

api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY est introuvable dans les variables d'environnement.")

client = Mistral(api_key=api_key)

# Le prompt système du dataset original (tel que présent dans data_sft_v1.0.0_reviewed.jsonl)
DATASET_SYSTEM_PROMPT = """Tu es un médecin urgentiste chargé de trier des situations cliniques.
Ton objectif est de décider entre deux actions :
- POSER UNE QUESTION si les informations sont insuffisantes ou ambiguës
- DONNER UN VERDICT MÉDICAL STRUCTURÉ si les informations sont suffisantes
Règles :
1. Tu dois toujours répondre au format JSON strict.
2. Si les informations sont insuffisantes, pose UNE seule question ciblée.
3. Si les informations sont suffisantes, donne une analyse médicale avec un niveau d'urgence.
4. L'urgence doit être strictement : "Haute", "Moyenne" ou "Faible".
5. Ne jamais inclure de texte hors JSON.
6. Sois concis, médicalement prudent et factuel.
Format attendu :
CAS QUESTION :
{
  "type": "question",
  "question": "...",
  "urgence": null,
  "analyse": null
}
CAS FINAL :
{
  "type": "final",
  "question": null,
  "urgence": "...",
  "analyse": "..."
}"""

# =========================
# HELPERS
# =========================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_new_example(cas_clinique, question):
    today = date.today().isoformat()
    
    # Format de l'assistant pour une question
    assistant_json = {
        "type": "question",
        "question": question.strip(),
        "urgence": None,
        "analyse": None
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": DATASET_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": cas_clinique.strip()
            },
            {
                "role": "assistant",
                # content assistant = dict JSON ({"type": "final"|"question", ...})
                "content": assistant_json
            }
        ],
        "metadata": str({
            "niveau_confiance": 4,
            "source": {
                "date_creation": today,
                "type_document": "Mistral-Synthetic-Questions"
            },
            "symptomes": "array([], dtype=object)",
            "tag_origine": "mistral_anglaise_Questions"
        })
    }


def build_generation_messages(batch_size):
    prompt = f"""Tu es un générateur de cas cliniques pour entraîner un modèle médical de triage.

OBJECTIF :
Produire UNIQUEMENT des cas où le modèle de triage devra POSER UNE QUESTION (car les informations fournies par le patient sont insuffisantes pour prendre une décision finale).
Chaque cas clinique (la plainte du patient) doit être réaliste, court (1 à 3 phrases max), et cliniquement pertinent.

CONTRAINTES GÉNÉRALES :
- Génère des situations très variées (âge, sexe, symptômes, contexte)
- Chaque cas doit être plausible en médecine d'urgence ou régulation médicale
- Introduis de l'ambiguïté réelle (un manque d'info critique : ex. la localisation exacte de la douleur, la durée, les symptômes associés, les antécédents, etc.)
- La "question" que le modèle de triage devrait poser doit être ciblée, utile médicalement et aider directement à lever l'ambiguïté critique. Pas de questions génériques.
- Ne JAMAIS produire de cas où toutes les informations sont déjà présentes.
- Ne JAMAIS ajouter d'explications.
- IMPORTANT : Les cas générés (cas clinique et question) doivent être rédigés en ANGLAIS.

DIVERSITÉ OBLIGATOIRE :
Varie fortement :
- symptômes (douleur, fièvre, neurologique, respiratoire, digestif, trauma, pédiatrique, psychiatrique…)
- profils patients (enfant, adulte, personne âgée, femme enceinte, sportif…)
- gravité implicite (potentiellement grave ou bénin incertain)
- contexte (domicile, sport, travail, nuit, voyage, accident…)

FORMAT STRICT JSON (LISTE) :
Tu dois produire une liste JSON de EXACTEMENT {batch_size} objets.

Chaque objet doit être EXACTEMENT sous cette forme :
{{
  "cas clinique": "short description of the patient in English",
  "question": "the targeted question to ask the patient in English"
}}

RÈGLES IMPORTANTES :
- JSON valide uniquement
- AUCUN texte hors JSON
- PAS de commentaires
- PAS de duplication de cas
- PAS de champs en plus ou en moins"""

    return [
        {"role": "user", "content": prompt}
    ]


def call_mistral_json(messages):
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=TEMPERATURE
            )

            content = response.choices[0].message.content

            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )

            return json.loads(content)

        except Exception as e:
            last_error = e
            time.sleep(1 + attempt)

    raise last_error


def extract_items_from_response(data):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    # Mistral peut renvoyer {"cases": [...]}, ou juste les clés
    for key in ["items", "cases", "examples", "data", "cas"]:
        value = data.get(key)
        if isinstance(value, list):
            return value
            
    # S'il y a une liste cachée dans d'autres clés
    for value in data.values():
        if isinstance(value, list):
            return value

    return []


def validate_generated_item(item):
    if not isinstance(item, dict):
        return None

    cas_clinique = item.get("cas clinique", item.get("cas_clinique"))
    question = item.get("question")

    if not isinstance(cas_clinique, str) or not cas_clinique.strip():
        return None

    if not isinstance(question, str) or not question.strip():
        return None

    return {
        "cas clinique": cas_clinique.strip(),
        "question": question.strip()
    }


def generate_synthetic_cases(n_cases, batch_size):
    generated = []

    while len(generated) < n_cases:
        current_batch = batch_size

        try:
            messages = build_generation_messages(current_batch)
            result = call_mistral_json(messages)
            items = extract_items_from_response(result)
            
            # Si le JSON parsing renvoie directement une liste
            if not items and isinstance(result, list):
                items = result

            valid_items = []
            for item in items:
                validated = validate_generated_item(item)
                if validated is not None:
                    valid_items.append(validated)

            if not valid_items:
                print("Aucun cas valide reçu sur ce batch.")
            else:
                for v in valid_items:
                    if len(generated) < n_cases:
                        generated.append(v)
                print(f"{len(generated)} / {n_cases} cas générés")

            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"Erreur génération batch: {e}")
            time.sleep(2)

    return generated


def main():
    print("Chargement du dataset existant...")
    if INPUT_DATASET_PATH.exists():
        existing_rows = load_jsonl(INPUT_DATASET_PATH)
        print(f"Dataset existant chargé: {len(existing_rows)} lignes")
    else:
        existing_rows = []
        print("Dataset existant non trouvé, création d'un nouveau.")

    print(f"Génération de {N_NEW_CASES} nouveaux cas avec des questions...")
    synthetic_cases = generate_synthetic_cases(N_NEW_CASES, BATCH_SIZE)

    synthetic_rows = [
        make_new_example(case["cas clinique"], case["question"])
        for case in synthetic_cases
    ]

    print(f"Nouveaux cas valides: {len(synthetic_rows)}")

    final_rows = existing_rows + synthetic_rows
    random.seed(SHUFFLE_SEED)
    random.shuffle(final_rows)

    save_jsonl(final_rows, OUTPUT_DATASET_PATH)

    print("\nTerminé")
    print(f"Ancien dataset : {len(existing_rows)}")
    print(f"Nouveaux cas   : {len(synthetic_rows)}")
    print(f"Total final    : {len(final_rows)}")
    print(f"Sauvegardé dans: {OUTPUT_DATASET_PATH}")


if __name__ == "__main__":
    main()
