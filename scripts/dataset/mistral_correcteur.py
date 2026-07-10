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

INPUT_FILE = DATASET_PATH / "data_sft_v1.0.0_reprompted.jsonl"
OUTPUT_FILE = DATASET_PATH / "data_sft_v1.0.0_reviewed.jsonl"
A_VERIFIER_FILE = DATASET_PATH / "data_sft_v1.0.0_a_verifier.jsonl"

VALID_URGENCE = {"Haute", "Moyenne", "Faible"}

MODEL_MEDIUM = "mistral-medium-latest"
MODEL_SMALL = "mistral-small-latest"

client = Mistral(api_key=API_KEY)


# ================================
# Prompts
# ================================
SYSTEM_PII = """Tu es un anonymiseur de données médicales.
Remplace dans le texte toutes les données personnelles identifiables par les balises :
- Prénom ou nom de personne → <PATIENT>
- Adresse, ville ou pays spécifique → <LIEU>
- Date précise (jour+mois+année) → <DATE>
- Téléphone ou email → <CONTACT>
Ne remplace PAS : âge, sexe, profession, symptômes, noms de maladies ou médicaments.
Retourne UNIQUEMENT le texte nettoyé, sans explication, sans JSON."""

SYSTEM_MEDIUM = """Tu es un correcteur expert de dataset pour un modèle de triage médical.
Tu dois analyser une conversation et produire UNE sortie JSON STRICTE.
Tu ne dois JAMAIS produire de texte hors JSON.
---
ÉTAPE 1 — INSPECTE CHAQUE MESSAGE [assistant] DANS LA CONVERSATION (sauf le dernier).
Dans le bloc "Conversation :", repère tous les tours "[assistant] ...".
Un message "[assistant]" intermédiaire est CASSÉ s'il est en texte libre et non un objet JSON.
Exemples de messages CASSÉS : "Je ne suis pas sûr...", "D'accord...", tout texte qui ne commence pas par '{'.
Si tu trouves AU MOINS UN message "[assistant]" intermédiaire cassé → retourne {"type":"a_verifier"} UNIQUEMENT, sans rien d'autre.
---
ÉTAPE 2 — Si et seulement si TOUS les messages "[assistant]" intermédiaires sont des JSON valides,
travaille sur la "Réponse à corriger (dernier message assistant)".
---
DÉFINITION DU TYPE — CRITIQUE :
Le type est déterminé par le CONTENU DE LA DERNIÈRE RÉPONSE DE L'ASSISTANT, pas par le message user.
- type = "question" : la DERNIÈRE réponse de l'assistant est une question de clarification au patient
- type = "final"    : la DERNIÈRE réponse de l'assistant est une analyse ou un verdict médical
---
CAS 1 — Dernière réponse exploitable (même si incorrecte) :
Retourne une réponse corrigée :
{"type":"question"|"final","question":string|null,"urgence":"Haute"|"Moyenne"|"Faible"|null,"analyse":string|null}
Règles :
1. type = "question" → remplir "question", "urgence"=null, "analyse"=null
2. type = "final"    → "question"=null, "urgence" obligatoire ("Haute"/"Moyenne"/"Faible"), remplir "analyse"
   - Si "urgence" est null → déduis-la depuis le contexte clinique
3. Corrige toute incohérence (urgence absente, champs inversés, mauvais type)
---
CAS 2 — Contenu non médical ou totalement inutilisable :
{"type":"reject"}
---
CAS 3 — Problème dans un message assistant INTERMÉDIAIRE (pas le dernier) :
{"type":"a_verifier"}
---
Niveaux d'urgence :
- Haute   : pronostic vital engagé (infarctus, AVC, sepsis, détresse respiratoire…)
- Moyenne : consultation rapide nécessaire (fracture, infection, douleur importante…)
- Faible  : peut attendre (rhume, entorse bénigne, information générale…)
---
RÈGLES STRICTES : JSON valide uniquement, aucun texte hors JSON, aucun champ supplémentaire."""

SYSTEM_SMALL = """Tu es un validateur de structure pour un dataset de triage médical.
Tu vérifies et corriges UNIQUEMENT la STRUCTURE de la dernière réponse de l'assistant.
Réponds UNIQUEMENT avec un JSON valide.
Format obligatoire :
{"type":"question"|"final","question":string|null,"urgence":"Haute"|"Moyenne"|"Faible"|null,"analyse":string|null}
---
DÉFINITION DU TYPE — CRITIQUE :
Le type est déterminé par le CONTENU DE LA DERNIÈRE RÉPONSE DE L'ASSISTANT, pas par le message user.
- type = "question" : la réponse de l'assistant EST une question de clarification → remplir "question"
- type = "final"    : la réponse de l'assistant EST une analyse médicale → remplir "analyse" + "urgence"
---
Corrige UNIQUEMENT :
- type mal attribué (question dans "analyse", analyse dans "question")
- "urgence" absente ou invalide sur un type "final"
- champs null/non-null mal placés
Ne modifie PAS le contenu médical, ne change pas le type sans raison structurelle évidente.
Si la structure est correcte → retourne-la telle quelle.
---
Aucun texte hors JSON. Toujours retourner un JSON complet."""


# ================================
# Nettoyage PII (user messages)
# ================================
def clean_user_pii(user_text, retries=2):
    """Passe légère Mistral Small pour anonymiser le PII résiduel dans un message user."""
    if not user_text or not isinstance(user_text, str):
        return user_text
    if user_text.strip().lower().startswith(("what are", "how is", "is there")):
        return user_text
    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model=MODEL_SMALL,
                messages=[
                    {"role": "system", "content": SYSTEM_PII},
                    {"role": "user", "content": user_text}
                ]
            )
            cleaned = response.choices[0].message.content.strip()
            if len(cleaned) >= len(user_text) * 0.4:
                return cleaned
        except Exception as e:
            print(f"Erreur PII cleaner (tentative {attempt + 1}) : {e}")
            time.sleep(2 ** attempt)
    return user_text 


# ================================
# Appel Mistral
# ================================
def strip_markdown_json(raw):
    """Retire les balises ```json ... ``` que certains modèles ajoutent."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def call_mistral(model, system_prompt, messages, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "system", "content": system_prompt}] + messages
            )
            raw = strip_markdown_json(response.choices[0].message.content)
            parsed = json.loads(raw, strict=False) 
            return parsed

        except json.JSONDecodeError as e:
            print(f"JSON invalide retourné par Mistral : {e} | raw: {raw[:80]}")
        except Exception as e:
            print(f"Erreur API Mistral (tentative {attempt + 1}) : {e}")
            time.sleep(2 ** attempt)

    return None


# ================================
# Validation Medium
# ================================
def validate_medium_response(response):
    if not isinstance(response, dict):
        return False

    msg_type = response.get("type")

    if msg_type == "reject":
        return "reject"

    if msg_type == "a_verifier":
        return "a_verifier"

    if msg_type == "question":
        if not response.get("question"):
            return False
        return True

    if msg_type == "final":
        if response.get("urgence") not in VALID_URGENCE:
            return False
        if not response.get("analyse"):
            return False
        return True

    return False


# ================================
# Validation Small
# ================================
def validate_small_response(response):
    if not isinstance(response, dict):
        return False

    msg_type = response.get("type")

    if msg_type == "question":
        if not response.get("question"):
            return False
        return True

    if msg_type == "final":
        if response.get("urgence") not in VALID_URGENCE:
            return False
        if not response.get("analyse"):
            return False
        return True

    return False


# ================================
# Construction messages pour Mistral
# ================================
def build_mistral_messages(messages):
    """
    Construit un message unique structuré pour Mistral.
    - Le prompt système du dataset est exclu (économie de tokens).
    - Toute la conversation (user + assistant intermédiaires) est incluse en contexte
      pour que Mistral puisse détecter les messages intermédiaires cassés.
    - La dernière réponse assistant est séparée comme "Réponse à corriger".
    """
    conversation_lines = []
    last_assistant_content = None
    last_assistant_idx = None

    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            last_assistant_idx = i

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            continue  

        content_str = json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)

        if role == "assistant" and i == last_assistant_idx:
            last_assistant_content = content_str
        else:
            conversation_lines.append(f"[{role}] {content_str}")

    combined = "Conversation :\n" + "\n".join(conversation_lines)
    if last_assistant_content:
        combined += f"\n\nRéponse à corriger (dernier message assistant) :\n{last_assistant_content}"

    return [{"role": "user", "content": combined}]


# ================================
# Traitement d'une ligne
# ================================
def process_line(data):
    messages = data.get("messages", [])

    # Identifier tous les messages assistant
    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]

    if not assistant_indices:
        return None, "skip"

    assistant_idx = assistant_indices[-1]
    assistant_content = messages[assistant_idx].get("content", {})

    for i in assistant_indices[:-1]:
        if not isinstance(messages[i].get("content"), dict):
            return data, "a_verifier"

    if assistant_content is None:
        return None, "skip"

    urgence = assistant_content.get("urgence") if isinstance(assistant_content, dict) else None
    mistral_messages = build_mistral_messages(messages)

    if urgence is None:
        # Mistral Medium
        response = call_mistral(MODEL_MEDIUM, SYSTEM_MEDIUM, mistral_messages)

        if response is None:
            return None, "skip"

        valid = validate_medium_response(response)

        if valid == "reject":
            print(f"Rejeté : {mistral_messages[-1].get('content', '')[:60]}")
            return None, "reject"

        if valid == "a_verifier":
            print(f"À vérifier : {mistral_messages[-1].get('content', '')[:60]}")
            return data, "a_verifier"

        if not valid:
            print(f"Réponse Medium invalide : {str(response)[:80]}")
            return None, "skip"

        messages[assistant_idx]["content"] = response
        data["messages"] = messages
        return data, "medium"

    else:
        # Mistral Small
        response = call_mistral(MODEL_SMALL, SYSTEM_SMALL, mistral_messages)

        if response is None:
            return None, "skip"

        if not validate_small_response(response):
            print(f"Réponse Small invalide : {str(response)[:80]}")
            return None, "skip"

        messages[assistant_idx]["content"] = response
        data["messages"] = messages
        return data, "small"


# ================================
# Traitement fichier
# ================================
def process_file():
    total = 0
    skipped = 0
    rejected = 0
    a_verifier = 0
    corrected_medium = 0
    corrected_small = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout, \
         open(A_VERIFIER_FILE, "w", encoding="utf-8") as fver:

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

            result, route = process_line(data)

            if route == "reject":
                rejected += 1
                continue

            if route == "a_verifier":
                fver.write(json.dumps(result, ensure_ascii=False) + "\n")
                a_verifier += 1
                continue

            if result is None:
                skipped += 1
                continue

            if route == "medium":
                corrected_medium += 1
            elif route == "small":
                corrected_small += 1

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            total += 1

    print(f"\n-- Dataset reviewé avec succès : {total} lignes écrites")
    print(f"\n-- Stats :")
    print(f"  Corrigés par Mistral Medium : {corrected_medium}")
    print(f"  Validés par Mistral Small   : {corrected_small}")
    print(f"  Rejetés                     : {rejected}")
    print(f"  À vérifier manuellement     : {a_verifier}")
    print(f"  Ignorés (erreurs)           : {skipped}")


# ================================
# RUN
# ================================
if __name__ == "__main__":
    process_file()