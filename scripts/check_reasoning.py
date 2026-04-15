import os
import json
import re
import time
import random
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from mistralai.client import Mistral

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "dataset_sft_final_v3.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "dataset_sft_final.jsonl"

MODEL_NAME = "mistral-small-latest"
TEMPERATURE = 0
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.3
SHUFFLE_SEED = 42

api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY est introuvable dans les variables d'environnement.")

client = Mistral(api_key=api_key)

tag_pattern = re.compile(r"'tag_origine'\s*:\s*'([^']+)'")

VALID_URGENCES = {"Haute", "Moyenne", "Faible"}


def extract_tag_origine(metadata):
    if isinstance(metadata, dict):
        return metadata.get("tag_origine")

    if isinstance(metadata, str):
        match = tag_pattern.search(metadata)
        if match:
            return match.group(1)

    return None


def get_last_user_and_assistant(messages):
    last_user = None
    last_assistant = None

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = msg.get("content", "")

        if role == "user":
            last_user = content
        elif role == "assistant":
            last_assistant = content

    return last_user, last_assistant


def strip_user_prefix(text):
    text = (text or "").strip()
    if text.startswith("### USER"):
        text = text[len("### USER"):].strip()
    return text


def parse_analyse_assistant_content(content):
    """
    Attend un contenu du type :
    ### ANALYSE
    {"urgence": "...", "analyse": "...", "symptome": [...]}
    """
    content = (content or "").strip()
    if not content.startswith("### ANALYSE"):
        return None

    body = content[len("### ANALYSE"):].strip()
    try:
        data = json.loads(body)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    return data


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


def normalize_urgence(value):
    value = str(value).strip()
    if value not in VALID_URGENCES:
        return "Faible"
    return value


def build_messages_for_mistral(user_text, analyse_text):
    return [
        {
            "role": "system",
            "content": (
                "Tu es un expert médical des urgences. "
                "Tu reçois une question utilisateur et une analyse médicale existante. "
                "Tu dois produire UNIQUEMENT un JSON valide avec exactement les champs : "
                "\"urgence\", \"symptomes\", \"analyse\". "
                "Règles : "
                "1) urgence doit être exactement l'une de : Haute, Moyenne, Faible. "
                "2) symptomes est une liste de symptômes pertinents, sans doublons. "
                "3) analyse doit être une reformulation plus concise, claire et cohérente de l'analyse. "
                "4) Ne renvoie aucun texte hors JSON."
            )
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user": user_text,
                    "analyse_existante": analyse_text
                },
                ensure_ascii=False
            )
        }
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


def update_last_assistant_analyse(messages, new_urgence, new_analyse, new_symptomes):
    """
    Remplace le DERNIER assistant ### ANALYSE par la nouvelle version.
    """
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue

        parsed = parse_analyse_assistant_content(msg.get("content", ""))
        if parsed is None:
            continue

        normalized = {
            "urgence": normalize_urgence(new_urgence),
            "analyse": str(new_analyse).strip(),
            "symptome": normalize_symptomes(new_symptomes),
        }

        messages[idx]["content"] = "### ANALYSE\n" + json.dumps(normalized, ensure_ascii=False)
        return True

    return False


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


def process_dataset(input_path, output_path):
    rows = load_jsonl(input_path)

    reasoning_rows = []
    non_reasoning_rows = []

    for row in rows:
        metadata = row.get("metadata", {})
        tag_origine = extract_tag_origine(metadata)

        if tag_origine == "Understanding":
            reasoning_rows.append(row)
        else:
            non_reasoning_rows.append(row)

    print(f"Total lignes: {len(rows)}")
    print(f"Understanding: {len(reasoning_rows)}")
    print(f"Non-Reasoning: {len(non_reasoning_rows)}")

    updated_reasoning_rows = []
    updated_count = 0
    skipped_count = 0
    error_count = 0

    for i, row in enumerate(reasoning_rows):
        try:
            messages = row.get("messages", [])
            user_content, assistant_content = get_last_user_and_assistant(messages)

            if not user_content or not assistant_content:
                skipped_count += 1
                updated_reasoning_rows.append(row)
                continue

            parsed_assistant = parse_analyse_assistant_content(assistant_content)
            if parsed_assistant is None:
                skipped_count += 1
                updated_reasoning_rows.append(row)
                continue

            user_text = strip_user_prefix(user_content)
            analyse_text = str(parsed_assistant.get("analyse", "")).strip()

            mistral_messages = build_messages_for_mistral(user_text, analyse_text)
            review = call_mistral_json(mistral_messages)

            new_urgence = normalize_urgence(review.get("urgence", parsed_assistant.get("urgence", "Faible")))
            new_symptomes = normalize_symptomes(review.get("symptomes", parsed_assistant.get("symptome", [])))
            new_analyse = str(review.get("analyse", analyse_text)).strip()
            if not new_analyse:
                new_analyse = analyse_text

            updated = update_last_assistant_analyse(
                messages=messages,
                new_urgence=new_urgence,
                new_analyse=new_analyse,
                new_symptomes=new_symptomes
            )

            if updated:
                updated_count += 1
            else:
                skipped_count += 1

            updated_reasoning_rows.append(row)

            if (i + 1) % 50 == 0:
                print(f"{i + 1} / {len(reasoning_rows)} cas Understanding traités")

            time.sleep(SLEEP_BETWEEN_CALLS)

        except Exception as e:
            error_count += 1
            print(f"Erreur sur Understanding index {i}: {e}")
            updated_reasoning_rows.append(row)

    final_rows = updated_reasoning_rows + non_reasoning_rows
    random.seed(SHUFFLE_SEED)
    random.shuffle(final_rows)

    save_jsonl(final_rows, output_path)

    print("\n-- Terminé")
    print(f"Understanding mis à jour: {updated_count}")
    print(f"Understanding skip: {skipped_count}")
    print(f"Erreurs: {error_count}")
    print(f"Dataset final sauvegardé dans: {output_path}")


process_dataset(INPUT_PATH, OUTPUT_PATH)