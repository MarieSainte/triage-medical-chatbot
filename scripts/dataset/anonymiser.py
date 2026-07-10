import sys
import json
import argparse
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# 1. Configuration des modèles spaCy
model_config = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "fr", "model_name": "fr_core_news_md"},
        {"lang_code": "en", "model_name": "en_core_web_md"},
    ],
}
nlp_engine = SpacyNlpEngine(models=model_config["models"])

# 2. Initialisation de l'analyzer
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.4)
anonymizer = AnonymizerEngine()

# --- PATTERNS ---
patterns = {
    "date_num": r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    "phone_fr": r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}",
    "us_address": r"\d{1,5}\s\w+(\s\w+)?\s(Street|St|Avenue|Ave|Road|Rd|Terrace|Ter|Drive|Dr)",
    "us_phone": r"(\d{3}-\d{3}-\d{4})|(\(\d{3}\)\s\d{3}-\d{4})|(\d{3}-\d{4})",
    "date_text": r"(\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})",
    "hospital": r"(\b\w+\s+(Hospital|Clinic|Centre|Medical Center|Infirmary|CHU|CH|Clinique|Hôpital|Hopital)\b(\s+\w+){0,3})",
    "medical_terms": r"(?i)\b(Freeman-Sheldon|Freeman Sheldon|Whistling face|Craniocarpotarsal|Down|Parkinson|Alzheimer|Crohn|Raynaud|Guillain-Barré|Lyme|expert|médecin|medecin|urgentiste|chirurgien|infirmier|psychologue|cardiologue|interne|docteur|user|assistant|role)\b"
}

# --- RECOGNIZERS ---
recognizers = [
    PatternRecognizer(supported_entity="DATE_TIME", patterns=[Pattern(name="date_fr", regex=patterns["date_num"], score=0.95)], supported_language="fr"),
    PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[Pattern(name="phone_fr", regex=patterns["phone_fr"], score=0.95)], supported_language="fr"),
    PatternRecognizer(supported_entity="DATE_TIME", patterns=[Pattern(name="date_text_fr", regex=patterns["date_text"], score=0.8)], supported_language="fr"),
    PatternRecognizer(supported_entity="LOCATION", patterns=[Pattern(name="us_address", regex=patterns["us_address"], score=0.7)], supported_language="en"),
    PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[Pattern(name="us_phone", regex=patterns["us_phone"], score=0.8)], supported_language="en"),
    PatternRecognizer(supported_entity="DATE_TIME", patterns=[Pattern(name="date_text_en", regex=patterns["date_text"], score=0.8)], supported_language="en"),
    PatternRecognizer(supported_entity="ORG", patterns=[Pattern(name="hospital_det", regex=patterns["hospital"], score=0.95)], supported_language="fr"),
    PatternRecognizer(supported_entity="ORG", patterns=[Pattern(name="hospital_det", regex=patterns["hospital"], score=0.95)], supported_language="en"),
    PatternRecognizer(supported_entity="MEDICAL_TERM", patterns=[Pattern(name="medical_terms", regex=patterns["medical_terms"], score=1.0)], supported_language="fr"),
    PatternRecognizer(supported_entity="MEDICAL_TERM", patterns=[Pattern(name="medical_terms", regex=patterns["medical_terms"], score=1.0)], supported_language="en"),
]

for rec in recognizers:
    analyzer.registry.add_recognizer(rec)

print("[OK] Moteur Anonymiseur Bilingue avec protection syndromes charge.")

# Liste de mots à ne JAMAIS anonymiser
mots_medicaux = [
    # Médicaments / unités
    "Aspirin", "Dose", "Mg", "Ml", "Doliprane", "Paracétamol", "Ibuprofène",
    "Ordonnance", "Matin", "Soir", "Midi", "Jour", "Semaine", "Heure",
    # Symptômes / contexte médical
    "Infection", "Toux", "Fièvre", "Douleur", "douleur", "Urgent", "Urgence", "Triage", "Clinique",
    # Rôles / structure dataset
    "user", "User", "assistant", "Assistant", "role", "Role",
    # Pronoms et articles fréquemment mal classifiés (FR) — minuscules ET majuscules
    "je", "Je", "il", "Il", "elle", "Elle", "on", "On", "nous", "Nous", "vous", "Vous",
    "quel", "Quel", "quelle", "Quelle", "quels", "Quels", "quelles", "Quelles",
    "le", "Le", "la", "La", "les", "Les", "l'",
    "un", "Un", "une", "Une", "des", "Des", "du", "Du",
    "mon", "Mon", "ma", "Ma", "mes", "Mes",
    "son", "Son", "sa", "Sa", "ses", "Ses",
    "leur", "Leur", "leurs", "Leurs",
    "ce", "Ce", "cet", "Cet", "cette", "Cette", "ces", "Ces",
    # Verbes courants mal classifiés
    "est", "sont", "a", "ont", "était", "avait", "être", "avoir",
    "prend", "prendre", "charge", "Charge",
    # Directions / côtés du corps (classifiés LOCATION par spaCy)
    "gauche", "droit", "droite",
    "supérieur", "inférieur", "supérieure", "inférieure",
    "antérieur", "postérieur", "antérieure", "postérieure",
    "proximal", "distal", "bilatéral", "bilatérale",
    # Termes corporels courants mal classifiés
    "poids", "Poids", "taille", "Taille",
    "bras", "jambe", "genou", "hanche", "épaule", "cheville",
    "formule", "Formule", "lignée", "Lignée",
    # Expressions multi-mots fréquemment mal détectées
    "du poids", "en charge", "La formule", "les anémies",
    # Mots anglais courants mal classifiés
    "I", "I'm", "My", "He", "She", "We", "They",
    "What", "Which", "How", "When", "Where",
    "pain", "fever", "weight", "left", "right",
]

def anonymiser_mixte_final(text):
    if not text or not isinstance(text, str):
        return text

    # spaCy ne détecte pas les noms précédés d'un saut de ligne.
    text_for_analysis = text.replace("\n", " ")

    entities_to_detect = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "MEDICAL_TERM"]

    res_fr = analyzer.analyze(text=text_for_analysis, entities=entities_to_detect, language='fr', score_threshold=0.8, allow_list=mots_medicaux)
    res_en = analyzer.analyze(text=text_for_analysis, entities=entities_to_detect, language='en', score_threshold=0.8, allow_list=mots_medicaux)
    
    all_results = res_fr + res_en
    
    # Stratégie de protection :
    # Si une entité (PERSON, etc.) contient ou est contenue dans un MEDICAL_TERM à 1.0, on la rejette.
    results_to_anonymize = []
    medical_ranges = [(r.start, r.end) for r in all_results if r.entity_type == "MEDICAL_TERM"]
    
    for res in all_results:
        if res.entity_type == "MEDICAL_TERM":
            continue
            
        is_protected = False
        for m_start, m_end in medical_ranges:
            if max(res.start, m_start) < min(res.end, m_end):
                is_protected = True
                break
        
        if not is_protected:
            results_to_anonymize.append(res)
    
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results_to_anonymize,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "<PATIENT>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LIEU>"}), 
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("mask", {"chars_to_mask": 12, "masking_char": "*", "from_end": False}),
        }
    )
    return anonymized.text

def anonymiser_general(content):
    """
    Anonymise soit une chaîne brute (messages user/system), soit un dict assistant.
    Le champ 'analyse' est intentionnellement exclu : c'est du raisonnement médical
    sans PII patient. Mistral correcteur gère le reste en aval.
    """
    if isinstance(content, str):
        return anonymiser_mixte_final(content)
    elif isinstance(content, dict):
        new_content = content.copy()
        # 'question' peut contenir une reformulation du cas patient → anonymiser
        # 'analyse' = raisonnement médical → pas de PII, on ne touche pas
        if "question" in new_content and isinstance(new_content["question"], str):
            new_content["question"] = anonymiser_mixte_final(new_content["question"])
        return new_content
    return content

def identifier_pii(text):
    """
    Identifie les entités PII dans le texte sans les anonymiser.
    Retourne une liste de dictionnaires avec le texte trouvé, le type et le score.
    """
    if not text or not isinstance(text, str):
        return []

    text_for_analysis = text.replace("\n", " ")

    entities_to_detect = ["PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "MEDICAL_TERM"]
    res_fr = analyzer.analyze(text=text_for_analysis, entities=entities_to_detect, language='fr', score_threshold=0.8, allow_list=mots_medicaux)
    res_en = analyzer.analyze(text=text_for_analysis, entities=entities_to_detect, language='en', score_threshold=0.8, allow_list=mots_medicaux)
    
    all_results = res_fr + res_en
    medical_ranges = [(r.start, r.end) for r in all_results if r.entity_type == "MEDICAL_TERM"]
    
    found_entities = []
    for res in all_results:
        if res.entity_type == "MEDICAL_TERM":
            continue
            
        is_protected = False
        for m_start, m_end in medical_ranges:
            if max(res.start, m_start) < min(res.end, m_end):
                is_protected = True
                break
        
        if not is_protected:
            found_entities.append({
                "text": text[res.start:res.end],
                "type": res.entity_type,
                "score": round(res.score, 2)
            })
            
    return found_entities

def anonymiser_conversation_sft(messages):
    new_messages = []
    for msg in messages:
        new_msg = msg.copy()
        if "content" in new_msg:
            new_msg["content"] = anonymiser_general(new_msg["content"])
        new_messages.append(new_msg)
    return new_messages

TAGS_BYPASS_ANONYMISATION = {"symptoms", "treatment"}

def get_tag_origine(data):
    """Extrait tag_origine depuis le champ metadata (stocké comme string repr)."""
    import re
    meta_raw = data.get("metadata", "")
    if not isinstance(meta_raw, str):
        return None
    m = re.search(r"'tag_origine':\s*'([^']+)'", meta_raw)
    return m.group(1) if m else None


def traiter_fichier_jsonl(input_path, output_path):
    """
    Lit un fichier JSONL, anonymise les conversations et écrit le résultat.
    Les lignes dont le tag_origine est dans TAGS_BYPASS_ANONYMISATION sont
    copiées telles quelles (pas de PII dans ces sources encyclopédiques).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f"[ERREUR] Le fichier d'entrée n'existe pas : {input_path}")
        return

    print(f"--- Début de l'anonymisation ---")
    print(f"Entrée : {input_path}")
    print(f"Sortie : {output_path}")

    total = 0
    bypassed = 0
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                tag = get_tag_origine(data)

                if tag in TAGS_BYPASS_ANONYMISATION:
                    bypassed += 1
                elif "messages" in data:
                    data["messages"] = anonymiser_conversation_sft(data["messages"])

                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                total += 1
                if total % 100 == 0:
                    print(f"Processé: {total} lignes...")
            except Exception as e:
                print(f"Erreur à la ligne {total+1}: {e}")

    print(f"✅ Terminé ! {total} lignes traitées (dont {bypassed} bypassed : {TAGS_BYPASS_ANONYMISATION}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymisation de fichiers JSONL médicaux.")
    parser.add_argument("input", nargs="?", help="Chemin du fichier JSONL d'entrée")
    parser.add_argument("output", nargs="?", help="Chemin du fichier JSONL de sortie")
    
    args = parser.parse_args()
    
    if not args.input:
        base_dir = Path(__file__).resolve().parent.parent
        args.input = base_dir / "data" / "data_versioned" / "data_sft_v1.0.0_cleaned.jsonl"
        if not args.output:
            args.output = base_dir / "data" / "data_versioned" / "data_sft_v1.0.0_anonymized.jsonl"
    
    if args.input and not args.output:
        path_in = Path(args.input)
        args.output = path_in.parent / f"{path_in.stem}_anonymized{path_in.suffix}"

    traiter_fichier_jsonl(args.input, args.output)