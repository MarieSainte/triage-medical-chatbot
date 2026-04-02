import dspy
trainset = [
    # CAS 1 : ANALYSE (Urgence)
    dspy.Example(
        symptomes="Douleur violente dans la poitrine, j'ai du mal à respirer et je transpire beaucoup.",
        reponse='### ANALYSE {"priorite": "URGENCE", "justification": "Signes évocateurs de syndrome coronaire aigu", "recommandation": "Appelez le 15 immédiatement, ne vous déplacez pas seul", "liste_des_symptomes": ["douleur poitrine", "dyspnée", "sueurs"]}'
    ).with_inputs('symptomes'),

    # CAS 2 : ASSISTANT (Manque d'infos)
    dspy.Example(
        symptomes="J'ai mal au ventre depuis ce matin.",
        reponse="### ASSISTANT La douleur est-elle localisée d'un côté précis et avez-vous de la fièvre ou des nausées ?"
    ).with_inputs('symptomes'),

    # CAS 3 : ANALYSE (Faible)
    dspy.Example(
        symptomes="Je me suis coupé avec une feuille de papier, ça saigne très peu mais ça pique.",
        reponse='### ANALYSE {"priorite": "FAIBLE", "justification": "Plaie superficielle sans signe de gravité", "recommandation": "Nettoyer à l\'eau et au savon, désinfecter et mettre un pansement", "liste_des_symptomes": ["coupure superficielle", "douleur légère"]}'
    ).with_inputs('symptomes'),

    # CAS 4 : ASSISTANT (Manque d'infos)
    dspy.Example(
        symptomes="J'ai des vertiges quand je me lève.",
        reponse="### ASSISTANT Est-ce que cela s'accompagne d'une perte d'équilibre, de sifflements d'oreilles ou de maux de tête ?"
    ).with_inputs('symptomes'),

    # CAS 5 : ANALYSE (Urgence)
    dspy.Example(
        symptomes="Ma jambe est toute rouge, gonflée et très chaude après une chirurgie la semaine dernière.",
        reponse='### ANALYSE {"priorite": "URGENCE", "justification": "Suspicion de thrombose veineuse profonde (phlébite)", "recommandation": "Rendez-vous aux urgences ou contactez votre chirurgien immédiatement", "liste_des_symptomes": ["oedème jambe", "chaleur locale", "érythème"]}'
    ).with_inputs('symptomes'),

    # CAS 6 : ASSISTANT (Manque d'infos)
    dspy.Example(
        symptomes="Je tousse beaucoup depuis deux jours.",
        reponse="### ASSISTANT Votre toux est-elle grasse ou sèche, et avez-vous des difficultés à reprendre votre souffle ?"
    ).with_inputs('symptomes'),

    # CAS 7 : ANALYSE (Relative)
    dspy.Example(
        symptomes="Je me suis tordu la cheville au foot, elle a doublé de volume et je ne peux plus poser le pied par terre.",
        reponse='### ANALYSE {"priorite": "RELATIVE", "justification": "Suspicion d\'entorse grave ou de fracture", "recommandation": "Appliquer de la glace, immobiliser et consulter pour une radio sous 12h", "liste_des_symptomes": ["traumatisme cheville", "oedème", "impotence fonctionnelle"]}'
    ).with_inputs('symptomes'),

    # CAS 8 : ASSISTANT (Manque d'infos)
    dspy.Example(
        symptomes="J'ai des plaques rouges sur le bras qui grattent.",
        reponse="### ASSISTANT Avez-vous mangé un nouvel aliment ou utilisé un nouveau produit, et ressentez-vous un gonflement du visage ?"
    ).with_inputs('symptomes'),

    # CAS 9 : ANALYSE (Faible)
    dspy.Example(
        symptomes="J'ai un gros rhume avec le nez qui coule et un peu mal à la gorge, pas de fièvre.",
        reponse='### ANALYSE {"priorite": "FAIBLE", "justification": "Infection virale bénigne des voies aériennes supérieures", "recommandation": "Lavage de nez au sérum phy et repos, consulter si fièvre supérieure à 38.5", "liste_des_symptomes": ["rhinorrhée", "maux de gorge"]}'
    ).with_inputs('symptomes'),

    # CAS 10 : ANALYSE (Urgence)
    dspy.Example(
        symptomes="Je vois flou d'un seul coup de l'œil droit et j'ai une partie du visage qui semble paralysée.",
        reponse='### ANALYSE {"priorite": "URGENCE", "justification": "Signes neurologiques suspects d\'AVC", "recommandation": "Appelez le 15 immédiatement. Notez l\'heure de début des symptômes.", "liste_des_symptomes": ["trouble vision", "paralysie faciale"]}'
    ).with_inputs('symptomes')
]