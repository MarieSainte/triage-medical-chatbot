import sys
import json
import re
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned"

INPUT_FILE = DATASET_PATH / "dataset_sft_qwen_final.jsonl"
OUTPUT_FILE = DATASET_PATH / "data_sft_v1.0.0_cleaned.jsonl"

VALID_URGENCE = {"Haute", "Moyenne", "Faible"}
URGENCE_FIELD_ALIASES = ["urgence", "priorite", "priority", "level", "niveau"]
PRIORITE_MAP = {
    1: "Faible",
    2: "Moyenne",
    3: "Haute",
    "1": "Faible",
    "2": "Moyenne",
    "3": "Haute"
}

REJECT_REASONS = {
    "json_invalide": 0,
    "pas_de_messages": 0,
    "build_output_none": 0,
    "moins_2_messages": 0,
    "ne_finit_pas_assistant": 0,
}


# ================================
# 1. Nettoyage des \n
# ================================
def fix_newlines(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"\n(?=\S)", "\n ", text)
    return text.strip()


# ================================
# 2. Détection JSON brut
# ================================
def is_json_content(text):
    text = text.strip()
    return text.startswith("{") or text.startswith("[")


# ================================
# 3. Détection vraie question de clarification
# ================================
def is_clarification_question(text):
    if len(text) > 300:
        return False
    if "?" not in text:
        return False
    return True


# ================================
# 4. Validation urgence
# ================================
def validate_urgence(value):
    if value in VALID_URGENCE:
        return value
    if value is not None:
        print(f"Urgence invalide ignorée : {repr(value)}")
    return None


def get_urgence_from_parsed(parsed):
    for field in URGENCE_FIELD_ALIASES:
        value = parsed.get(field)
        if value is not None:
            if value in PRIORITE_MAP:
                return PRIORITE_MAP[value]
            return validate_urgence(value)
    return None


# ================================
# 5. Détection type
# ================================
def detect_type(content):
    has_analyse = "### ANALYSE" in content
    has_assistant = "### ASSISTANT" in content
    if has_analyse and has_assistant:
        return "mixed"
    if has_analyse:
        return "final"
    if has_assistant:
        return "question"
    return None


# ================================
# 6. Extraction JSON robuste
# ================================
def fix_json_escapes(text):
    return re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)


def safe_json_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON invalide : {e} | texte: {text[:80]}")
        return None


def extract_json_from_content(content):
    marker = "### ANALYSE"
    idx = content.find(marker)
    if idx == -1:
        return None

    start = content.find("{", idx)
    if start == -1:
        return None

    json_str = fix_json_escapes(content[start:])

    try:
        obj, _ = json.JSONDecoder().raw_decode(json_str)
        return obj
    except json.JSONDecodeError as e:
        print(f"JSON invalide : {e} | texte: {json_str[:80]}")
        return None


# ================================
# 7. Format uniforme
# ================================
def build_output(msg_type, content):
    if msg_type == "question":
        question_text = fix_newlines(content.replace("### ASSISTANT", "").strip())

        if is_json_content(question_text):
            parsed = safe_json_parse(question_text)
            if isinstance(parsed, dict):
                return {
                    "type": "final",
                    "question": None,
                    "urgence": get_urgence_from_parsed(parsed),
                    "analyse": fix_newlines(parsed.get("analyse"))
                }

        if not is_clarification_question(question_text):
            return {
                "type": "final",
                "question": None,
                "urgence": None,
                "analyse": question_text
            }

        return {
            "type": "question",
            "question": question_text,
            "urgence": None,
            "analyse": None
        }

    elif msg_type == "final":
        parsed = extract_json_from_content(content)

        if isinstance(parsed, dict):
            return {
                "type": "final",
                "question": None,
                "urgence": get_urgence_from_parsed(parsed),
                "analyse": fix_newlines(parsed.get("analyse"))
            }

        return None

    elif msg_type == "mixed":
        parsed = extract_json_from_content(content)

        return {
            "type": "final",
            "question": None,
            "urgence": get_urgence_from_parsed(parsed) if parsed else None,
            "analyse": fix_newlines(parsed.get("analyse")) if parsed else None
        }


# ================================
# 8. Traitement messages
# ================================
def process_messages(messages):
    cleaned = []
    last_role = None

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            msg["content"] = fix_newlines(msg.get("content", ""))
            cleaned.append(msg)
            last_role = role
            continue

        if role in ("user", "assistant") and last_role == role:
            continue

        if role == "user":
            content = fix_newlines(msg.get("content", ""))
            content = content.replace("### USER", "").strip()
            if "### ASSISTANT" in content:
                content = content.split("### ASSISTANT")[0].strip()
            if not content:
                continue
            cleaned.append({"role": "user", "content": content})
            last_role = role

        elif role == "assistant":
            content = msg.get("content", "")
            msg_type = detect_type(content)

            if not msg_type:
                continue

            structured = build_output(msg_type, content)
            if structured is None:
                REJECT_REASONS["build_output_none"] += 1
                continue

            cleaned.append({"role": "assistant", "content": structured})
            last_role = role

            if msg_type in ("final", "mixed"):
                break

    non_system = [m for m in cleaned if m["role"] != "system"]
    if len(non_system) < 2:
        REJECT_REASONS["moins_2_messages"] += 1
        return None

    if cleaned[-1]["role"] != "assistant":
        REJECT_REASONS["ne_finit_pas_assistant"] += 1
        return None

    return cleaned


# ================================
# 9. Traitement fichier
# ================================
def process_file():
    total = 0
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Ligne JSON invalide ignorée : {e}")
                REJECT_REASONS["json_invalide"] += 1
                skipped += 1
                continue

            messages = data.get("messages")
            if not messages:
                REJECT_REASONS["pas_de_messages"] += 1
                skipped += 1
                continue

            new_messages = process_messages(messages)
            if not new_messages:
                skipped += 1
                continue

            data["messages"] = new_messages
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            total += 1

    print(f"\n-- Dataset transformé avec succès : {total} lignes écrites, {skipped} ignorées.")
    print(f"\n-- Raisons de rejet :")
    for reason, count in REJECT_REASONS.items():
        print(f"  {reason} : {count}")


# ================================
# RUN
# ================================
if __name__ == "__main__":
    process_file()