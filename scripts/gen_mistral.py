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

INPUT_DATASET_PATH = BASE_DIR / "data" / "dataset_sft_final_mistral_reasoning.jsonl"
OUTPUT_DATASET_PATH = BASE_DIR / "data" / "dataset_sft_final_rework.jsonl"

N_NEW_CASES = 500
BATCH_SIZE = 2

MODEL_NAME = "mistral-small-latest"
TEMPERATURE = 0.7
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.3
SHUFFLE_SEED = 42

NEW_SYSTEM_PROMPT = (
    "Tu es un médecin urgentiste et régulateur expert. "
    "Si les informations sont insuffisantes, réponds avec ### ASSISTANT et pose une question ciblée. "
    "Si les informations sont suffisantes, réponds avec ### ANALYSE sous forme JSON avec exactement les champs "
    "\"urgence\", \"analyse\" et \"symptome\". "
    "Le champ \"urgence\" doit obligatoirement être l'une des valeurs suivantes : \"Haute\", \"Moyenne\", \"Faible\". "
    "Pour les questions médicales théoriques ou factuelles, réponds avec ### ASSISTANT de manière concise et rigoureuse. "
    "Réponds dans la même langue que celle utilisée par l'utilisateur."
)

VALID_URGENCES = {"Haute", "Moyenne", "Faible"}

api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY est introuvable dans les variables d'environnement.")

client = Mistral(api_key=api_key)

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


def normalize_urgence(value):
    value = str(value).strip()
    if value not in VALID_URGENCES:
        return "Moyenne"
    return value


def normalize_symptomes(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    cleaned = []
    seen = set()
    for x in value:
        x = str(x).strip()
        if not x:
            continue
        k = x.lower()
        if k not in seen:
            seen.add(k)
            cleaned.append(x)
    return cleaned


def make_new_example(user_text, analyse_json):
    today = date.today().isoformat()
    symptomes = normalize_symptomes(
        analyse_json.get("symptome", analyse_json.get("symptomes", []))
    )
    normalized_json = {
        "urgence": normalize_urgence(analyse_json.get("urgence")),
        "analyse": str(analyse_json.get("analyse", "")).strip(),
        "symptome":symptomes,
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": NEW_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"### USER\n{user_text.strip()}"
            },
            {
                "role": "assistant",
                "content": "### ANALYSE\n" + json.dumps(normalized_json, ensure_ascii=False)
            }
        ],
        "metadata": {
            "niveau_confiance": 4,
            "source": {
                "date_creation": today,
                "type_document": "Mistral-Synthetic-Generation"
            },
            "symptomes": symptomes,
            "tag_origine": "mistral_anglaise_Direct"
        }
    }


def build_generation_messages(batch_size):
    system_prompt = (
        "You are generating synthetic English emergency medicine training examples for a medical triage dataset. "
        "Generate only direct cases where the assistant has enough information to provide a final triage analysis immediately. "
        "Do not generate cases that require asking clarifying questions. "
        "Create a balanced mix of domains such as trauma, pediatrics, cardiology, neurology, infectious disease, respiratory, abdominal pain, toxicology, and general emergency complaints. "
        "Ensure each case is significantly different from the others"
        "Return ONLY valid JSON."
    )

    user_prompt = f"""
        Generate exactly {batch_size} synthetic English medical cases.

        Return a JSON array.
        Each item must have exactly these keys:
        - "user": query describing a complete case with enough information for direct triage
        - "assistant": an object with exactly these keys:
        - "urgence" must be EXACTLY one of: "Haute", "Moyenne", "Faible"
        - "analyse" must be:
            - concise
            - medically coherent
            - written in English
            - include:
                1) a short clinical justification for the urgency level
                2) a short actionable recommendation for the physician managing the patient
        - "symptome": a list of relevant symptoms in English

        Rules:
        - All cases must be in English
        - All assistant analyses must be direct final analyses, not questions
        - Keep user cases realistic and varied
        - Keep the analysis concise
        - No markdown
        - No text outside JSON
        """.strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
    """
    Supporte soit :
    {"items": [...]}
    soit
    {"cases": [...]}
    soit
    {"examples": [...]}
    soit
    {"data": [...]}
    """
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        return []

    for key in ["items", "cases", "examples", "data"]:
        value = data.get(key)
        if isinstance(value, list):
            return value

    return []


def validate_generated_item(item):
    if not isinstance(item, dict):
        return None

    user_text = item.get("user")
    assistant = item.get("assistant")

    if not isinstance(user_text, str) or not user_text.strip():
        return None

    if not isinstance(assistant, dict):
        return None

    urgence = normalize_urgence(assistant.get("urgence"))
    analyse = str(assistant.get("analyse", "")).strip()
    symptome = normalize_symptomes(assistant.get("symptome", assistant.get("symptomes", [])))

    if not analyse:
        return None

    return {
        "user": user_text.strip(),
        "assistant": {
            "urgence": urgence,
            "analyse": analyse,
            "symptome": symptome
        }
    }


def generate_synthetic_cases(n_cases, batch_size):
    generated = []

    while len(generated) < n_cases:
        remaining = n_cases - len(generated)
        current_batch = min(batch_size, remaining)

        try:
            messages = build_generation_messages(current_batch)
            result = call_mistral_json(messages)
            items = extract_items_from_response(result)

            valid_items = []
            for item in items:
                validated = validate_generated_item(item)
                if validated is not None:
                    valid_items.append(validated)

            if not valid_items:
                print("Aucun cas valide reçu sur ce batch.")
            else:
                generated.extend(valid_items)
                print(f"{len(generated)} / {n_cases} cas générés")

            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            print(f"Erreur génération batch: {e}")
            time.sleep(2)

    return generated[:n_cases]


def main():
    print("Chargement du dataset existant...")
    existing_rows = load_jsonl(INPUT_DATASET_PATH)
    print(f"Dataset existant chargé: {len(existing_rows)} lignes")

    print(f"Génération de {N_NEW_CASES} nouveaux cas anglais...")
    synthetic_cases = generate_synthetic_cases(N_NEW_CASES, BATCH_SIZE)

    synthetic_rows = [
        make_new_example(case["user"], case["assistant"])
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



main()