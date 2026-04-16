
DATASET = [
    {
        "input": "J'ai une douleur violente dans la poitrine depuis 20 minutes, je transpire beaucoup et j'ai du mal à respirer.",
        "expected_type": "final",
        "expected_urgence": "Haute",
        "expected_analyse": "Syndrome coronaire aigu suspecté",
        "expected_question": None
    },
    {
        "input": "Mon mari de 62 ans a soudainement perdu la parole, sa bouche est déviée et il ne peut plus lever le bras droit.",
        "expected_type": "final",
        "expected_urgence": "Haute",
        "expected_analyse": "Suspicion d'AVC",
        "expected_question": None
    },
    {
        "input": "J'ai mangé des cacahuètes il y a 10 minutes. Mon visage gonfle, ma gorge se resserre et j'ai des plaques rouges partout.",
        "expected_type": "final",
        "expected_urgence": "Haute",
        "expected_analyse": "Choc anaphylactique probable",
        "expected_question": None
    },
    {
        "input": "Je suis tombé de vélo, mon poignet est très gonflé, j'ai très mal et je ne peux plus le bouger.",
        "expected_type": "final",
        "expected_urgence": "Moyenne",
        "expected_analyse": "Suspicion de fracture",
        "expected_question": None
    },
    {
        "input": "J'ai de la fièvre à 39°C depuis deux jours, des frissons, mal dans le dos côté droit et des brûlures en urinant.",
        "expected_type": "final",
        "expected_urgence": "Moyenne",
        "expected_analyse": "Suspicion de pyélonéphrite",
        "expected_question": None
    },
    {
        "input": "J'ai le nez qui coule, un peu mal à la gorge et je suis fatigué depuis hier. Pas de fièvre.",
        "expected_type": "final",
        "expected_urgence": "Faible",
        "expected_analyse": "Infection virale bénigne",
        "expected_question": None
    },
    {
        "input": "Je me suis tordu la cheville en descendant un escalier, ça fait un peu mal mais je peux encore marcher.",
        "expected_type": "final",
        "expected_urgence": "Faible",
        "expected_analyse": "Entorse bénigne",
        "expected_question": None
    },
    {
        "input": "J'ai mal au ventre depuis ce matin.",
        "expected_type": "question",
        "expected_urgence": None,
        "expected_analyse": None,
        "expected_question": "Demander localisation ou symptômes associés"
    },
    {
        "input": "Je me sens très fatigué depuis quelques jours.",
        "expected_type": "question",
        "expected_urgence": None,
        "expected_analyse": None,
        "expected_question": "Demander fièvre ou essoufflement"
    },
    {
        "input": "J'ai des maux de tête.",
        "expected_type": "question",
        "expected_urgence": None,
        "expected_analyse": None,
        "expected_question": "Demander mode d'apparition ou signes associés"
    }
]