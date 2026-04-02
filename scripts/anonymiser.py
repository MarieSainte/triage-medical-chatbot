
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# 1. On définit la configuration attendue par Presidio
model_config = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "fr", "model_name": "fr_core_news_md"},
        {"lang_code": "en", "model_name": "en_core_web_md"},
    ],
}

# 2. On crée l'engine proprement avec cette config
# Cela évite l'erreur "lang_code is missing"
nlp_engine = SpacyNlpEngine(models=model_config["models"])

# 3. Initialisation de l'analyzer
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, default_score_threshold=0.9)
anonymizer = AnonymizerEngine()

print("✅ Moteur Presidio Bilingue initialisé avec succès.")

# --- PATTERNS ---
patterns = {
    "date_num": r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    "phone_fr": r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}",
    "us_address": r"\d{1,5}\s\w+(\s\w+)?\s(Street|St|Avenue|Ave|Road|Rd|Terrace|Ter|Drive|Dr)",
    "us_phone": r"(\d{3}-\d{3}-\d{4})|(\(\d{3}\)\s\d{3}-\d{4})|(\d{3}-\d{4})",
    "date_text": r"(\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})",
    "hospital": r"(\b\w+\s+(Hospital|Clinic|Centre|Medical Center|Infirmary|CHU|CH|Clinique|Hôpital|Hopital)\b(\s+\w+){0,3})"
}

# --- ENREGISTREMENT ---
recognizers = [
    # Français
    PatternRecognizer(supported_entity="DATE_TIME", patterns=[Pattern(name="date_fr", regex=patterns["date_num"], score=0.95)], supported_language="fr"),
    PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[Pattern(name="phone_fr", regex=patterns["phone_fr"], score=0.95)], supported_language="fr"),
    PatternRecognizer(supported_entity="DATE_TIME", patterns=[Pattern(name="date_text_fr", regex=patterns["date_text"], score=0.8)], supported_language="fr"),
    
    # Anglais
    PatternRecognizer(supported_entity="LOCATION", patterns=[Pattern(name="us_address", regex=patterns["us_address"], score=0.7)], supported_language="en"),
    PatternRecognizer(supported_entity="PHONE_NUMBER", patterns=[Pattern(name="us_phone", regex=patterns["us_phone"], score=0.8)], supported_language="en"),
    PatternRecognizer(supported_entity="DATE_TIME", patterns=[Pattern(name="date_text_en", regex=patterns["date_text"], score=0.8)], supported_language="en"),
    
    # Universel (Hôpitaux) - On l'ajoute pour les deux langues
    PatternRecognizer(supported_entity="ORG", patterns=[Pattern(name="hospital_det", regex=patterns["hospital"], score=0.95)], supported_language="fr"),
    PatternRecognizer(supported_entity="ORG", patterns=[Pattern(name="hospital_det", regex=patterns["hospital"], score=0.95)], supported_language="en"),
]

for rec in recognizers:
    analyzer.registry.add_recognizer(rec)

print(f"✅ Tous les détecteurs (Dates, Tél, Adresses, Hôpitaux) sont chargés.")

mots_medicaux = [
    # Médicaments et dosages
    "Aspirin", "Rivaroxaban", "Clopidogrel", "Dose", "Daily", "Mg", "Mcg", "G", "Ml",
    "Pénicilline", "Amoxicilline", "Doliprane", "Paracétamol", "Ibuprofène", "Insuline",
    "Ventoline", "Bouffées", "Inhalateur", "Posologie", "Traitement", "Ordonnance",
    
    # Temps et Fréquence
    "Matin", "Soir", "Midi", "Jour", "Semaine", "Mois", "An", "Ans", "Heure", "Heures",
    "Quotidien", "Hebdomadaire", "Chaque", "Pendant", "Durant",
    
    # Anatomie et Symptômes
    "Infection", "Voies", "Respiratoires", "Laryngite", "Pharyngite", "Toux", "Fièvre",
    "Douleur", "Thoracique", "Abdominale", "Céphalée", "Dyspnée", "Pouls", "Tension",
    "Artérielle", "Cœur", "Poumons", "Foie", "Reins", "Rénale", "Glomérulaire",
    
    # Termes de Triage/Urgence
    "Urgent", "Urgence", "Régulateur", "Triage", "Signes", "Gravité", "Flags", "Red",
    "Stable", "Instable", "Pronostic", "Diagnostic", "Examen", "Clinique",
    
    # Verbes d'instruction (pour protéger ton prompt)
    "Analyse", "Propose", "Reste", "Aide", "Expert", "Conseille", "Oriente", "Évalue"
]

def anonymiser_mixte_final(text):
    if not text or not isinstance(text, str):
        return text
    
    entities_to_detect = ["PERSON", "LOCATION", "EMAIL_ADDRESS","PHONE_NUMBER"]
    
    # On force un score minimum (0.7) pour éviter de masquer les mots techniques
    res_fr = analyzer.analyze(text=text, entities=entities_to_detect, language='fr', score_threshold=0.9, allow_list=mots_medicaux)
    res_en = analyzer.analyze(text=text, entities=entities_to_detect, language='en', score_threshold=0.9, allow_list=mots_medicaux)
    
    all_results = res_fr + res_en
    
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=all_results,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "<PATIENT>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LIEU>"}), 
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("mask", {"chars_to_mask": 12, "masking_char": "*", "from_end": False}),
        }
    )
    return anonymized.text

def anonymiser_conversation_sft(messages):
    if not messages or not isinstance(messages, list):
        return messages
    
    new_messages = []
    
    NOUVEAU_PROMPT = (
        "Tu es un médecin urgentiste et régulateur expert. Ton rôle est d'analyser "
        "rapidement des situations cliniques pour effectuer un triage précis "
        "(déterminer le niveau d'urgence et extraire les symptômes clés), ou de "
        "répondre avec rigueur et clarté aux questions médicales théoriques. "
        "Ton raisonnement et ton verdict médical final doivent être structurés "
        "sous la balise ### ANALYSE. Si tu dois poser une question pour établir "
        "un diagnostic ou répondre directement à une question médicale, "
        "utilise la balise ### ASSISTANT."
    )
    
    for i, msg in enumerate(messages):
        new_msg = msg.copy()
        content = new_msg.get("content", "")
        
        # SI C'EST LE PREMIER MESSAGE (celui qui contient l'instruction polluée)
        if i == 0 and "User:" in content:
            # On coupe tout ce qui est avant "User:" pour supprimer l'ancien prompt
            # et on injecte le NOUVEAU_PROMPT à la place
            corps_message = content[content.find("User:"):]
            new_msg["content"] = f"{NOUVEAU_PROMPT}\n\n{corps_message}"
        
        # POUR TOUS LES MESSAGES (y compris le premier après nettoyage)
        # On lance l'anonymisation douce (seulement PERSON et seuil 0.9)
        if "content" in new_msg:
            new_msg["content"] = anonymiser_mixte_final(new_msg["content"])
            
        new_messages.append(new_msg)
            
    return new_messages