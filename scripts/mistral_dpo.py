# -*- coding: utf-8 -*-
import sys
import re
import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from mistralai.client import Mistral

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

# ================================
# Config
# ================================
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned"

INPUT_FILE  = DATASET_PATH / "data_dpo_cleaned.jsonl"
OUTPUT_FILE = DATASET_PATH / "data_dpo_structured.jsonl"
SKIPPED_FILE = DATASET_PATH / "data_dpo_skipped.jsonl"

VALID_URGENCE = {"Haute", "Moyenne", "Faible"}
MODEL = "mistral-medium-latest"

client = Mistral(api_key=API_KEY)

# ================================
# Prompt système Mistral
# ================================
SYSTEM_STRUCTURER = """Tu es un expert en triage médical. On te donne une question médicale et une réponse en texte libre.
Tu dois structurer cette réponse au format JSON strict selon les règles suivantes.

DÉFINITION DU TYPE — CRITIQUE :
- type = "question" : la réponse EST une question de clarification adressée au patient (demande d'information supplémentaire)
- type = "final"    : la réponse EST une analyse ou un verdict médical (même partiel)

FORMAT OBLIGATOIRE :
CAS QUESTION :
{"type": "question", "question": "...", "urgence": null, "analyse": null}

CAS FINAL :
{"type": "final", "question": null, "urgence": "Haute"|"Moyenne"|"Faible", "analyse": "..."}

Règles :
1. Retourne UNIQUEMENT le JSON, sans aucun texte hors JSON.
2. type = "question" → remplir "question" avec la question de clarification, urgence=null, analyse=null.
3. type = "final" → "question"=null, "urgence" obligatoire ("Haute"/"Moyenne"/"Faible"), remplir "analyse" avec le contenu médical.
4. Si la réponse est clairement une analyse médicale mais sans urgence explicite → déduis-la depuis le contexte clinique.
5. Si le contenu est totalement non médical ou inutilisable → {"type": "reject"}.

Niveaux d'urgence :
- Haute   : pronostic vital engagé (infarctus, AVC, sepsis, détresse respiratoire, hémorragie grave...)
- Moyenne : consultation rapide nécessaire (fracture, infection, douleur importante, grossesse à risque...)
- Faible  : peut attendre (rhume, entorse bénigne, information générale, prévention...)"""


# ================================
# Helpers
# ================================
def strip_markdown_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def call_mistral(user_question, response_text, retries=3):
    user_msg = (
        f"Question médicale : {user_question}\n\n"
        f"Réponse à structurer : {response_text}"
    )
    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_STRUCTURER},
                    {"role": "user",   "content": user_msg},
                ]
            )
            raw = strip_markdown_json(response.choices[0].message.content)
            parsed = json.loads(raw, strict=False)
            return parsed
        except json.JSONDecodeError as e:
            print(f"  JSON invalide : {e} | raw: {raw[:80]}")
        except Exception as e:
            print(f"  Erreur API (tentative {attempt + 1}) : {e}")
            time.sleep(2 ** attempt)
    return None


def validate_response(resp):
    if not isinstance(resp, dict):
        return False
    t = resp.get("type")
    if t == "reject":
        return "reject"
    if t == "question":
        return bool(resp.get("question"))
    if t == "final":
        if resp.get("urgence") not in VALID_URGENCE:
            return False
        if not resp.get("analyse"):
            return False
        return True
    return False


# ================================
# Traitement principal
# ================================
def process_file():
    total = 0
    skipped = 0
    rejected = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout, \
         open(SKIPPED_FILE, "w", encoding="utf-8") as fskip:

        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Ligne {line_num} JSON invalide : {e}")
                skipped += 1
                continue

            prompt = data.get("prompt", [])
            chosen_text = data.get("chosen_text", "")
            rejected_text = data.get("rejected_text", "")

            user_question = ""
            for msg in prompt:
                if msg.get("role") == "user":
                    user_question = msg.get("content", "")
                    break

            if not user_question or not chosen_text or not rejected_text:
                skipped += 1
                fskip.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            chosen_resp = call_mistral(user_question, chosen_text)
            chosen_valid = validate_response(chosen_resp)

            if chosen_valid == "reject" or not chosen_valid:
                print(f"  Ligne {line_num} : chosen rejeté/invalide")
                rejected += 1
                fskip.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            rejected_resp = call_mistral(user_question, rejected_text)
            rejected_valid = validate_response(rejected_resp)

            if rejected_valid == "reject" or not rejected_valid:
                print(f"  Ligne {line_num} : rejected rejeté/invalide")
                rejected += 1
                fskip.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            out = {
                "prompt": prompt,
                "chosen": [{"role": "assistant", "content": chosen_resp}],
                "rejected": [{"role": "assistant", "content": rejected_resp}],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            total += 1

            if total % 20 == 0:
                print(f"  Progression : {total} lignes traitees (ligne {line_num}/{364})")

    print(f"\nOK : {total} lignes ecrites")
    print(f"Rejetes/invalides : {rejected}")
    print(f"Ignores (erreurs) : {skipped}")
    print(f"Sortie : {OUTPUT_FILE}")


if __name__ == "__main__":
    process_file()
