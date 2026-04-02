import dspy

class TriageMedical(dspy.Signature):
    """Tu es un médecin régulateur. Évalue le cas clinique et fournis un triage structuré."""
    
    cas_clinique = dspy.InputField(desc="Le dossier patient brut avec ses constantes et antécédents")
    question = dspy.InputField(desc="La demande médicale ou le motif de recours")
    
    priorite = dspy.OutputField(desc="Niveau d'urgence: Haute, Modérée, ou Faible")
    analyse = dspy.OutputField(desc="Analyse médicale justifiant la priorité")
    symptomes_json = dspy.OutputField(desc="Liste stricte des symptômes au format JSON ['s1', 's2']")