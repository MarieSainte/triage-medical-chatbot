import sys
import re
import json
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned"

INPUT_FILE  = DATASET_PATH / "data_dpo.jsonl"
OUTPUT_FILE = DATASET_PATH / "data_dpo_cleaned.jsonl"

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


def extract_user_from_prompt(prompt: str) -> str:
    """Extrait la question user du prompt monolithique (après 'User:')."""
    match = re.search(r"\nUser:\s*(.+)", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    parts = prompt.split("\n\n", 1)
    return parts[-1].strip()


def clean_response(text: str) -> str:
    """Retire le préfixe ### ASSISTANT et nettoie les espaces."""
    text = re.sub(r"^###\s*ASSISTANT\s*\n*", "", text.strip())
    return text.strip()


def process_file():
    total = 0
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                prompt_raw  = data.get("prompt", "")
                chosen_raw  = data.get("chosen", "")
                rejected_raw = data.get("rejected", "")

                user_text    = extract_user_from_prompt(prompt_raw)
                chosen_text  = clean_response(chosen_raw)
                rejected_text = clean_response(rejected_raw)

                if not user_text or not chosen_text or not rejected_text:
                    skipped += 1
                    continue

                out = {
                    "prompt": [
                        {"role": "system", "content": NEW_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_text},
                    ],
                    "chosen_text":   chosen_text,
                    "rejected_text": rejected_text,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                total += 1

            except Exception as e:
                print(f"Erreur ligne {total + skipped + 1} : {e}")
                skipped += 1

    print(f"OK : {total} lignes ecrites, {skipped} ignorees.")
    print(f"Sortie : {OUTPUT_FILE}")


if __name__ == "__main__":
    process_file()
