import sys
import json
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned"

INPUT_FILE = DATASET_PATH / "data_sft_v1.0.0_anonymized.jsonl"
OUTPUT_FILE = DATASET_PATH / "data_sft_v1.0.0_reprompted.jsonl"

NEW_SYSTEM_PROMPT = """Tu es un médecin urgentiste chargé de trier des situations cliniques.
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


# ================================
# Remplacement du prompt système
# ================================
def replace_system_prompt(messages):
    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = NEW_SYSTEM_PROMPT
            return messages

    messages.insert(0, {
        "role": "system",
        "content": NEW_SYSTEM_PROMPT
    })
    return messages


# ================================
# Traitement fichier
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
                skipped += 1
                continue

            messages = data.get("messages")
            if not messages:
                skipped += 1
                continue

            data["messages"] = replace_system_prompt(messages)
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            total += 1

    print(f"Prompt remplacé avec succès : {total} lignes écrites, {skipped} ignorées.")


# ================================
# RUN
# ================================
if __name__ == "__main__":
    process_file()