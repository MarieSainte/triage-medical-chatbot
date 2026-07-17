
"""
Usage :
    uv run scripts/generate_dspy_prompts.py --adapter sft
    uv run scripts/generate_dspy_prompts.py --adapter dpo

Sortie :
    data/dspy_optimized_triage_<adapter>.json   <- programme compile
    data/dspy_prompt_preview_<adapter>.txt       <- apercu lisible
"""

import sys
import json
import argparse
from pathlib import Path
import unsloth
from unsloth import FastLanguageModel
import torch

# ==========================================
# 1. ARGUMENTS CLI
# ==========================================
parser = argparse.ArgumentParser(description="Optimisation DSPy des prompts triage")
parser.add_argument(
    "--adapter",
    choices=["sft", "dpo"],
    default="sft",
    help="Source modèle : 'sft' charge le LoRA SFT, 'dpo' charge le LoRA DPO."
)
parser.add_argument(
    "--adapter-path",
    default=None,
    help="Chemin explicite vers le modèle à charger (prioritaire sur le mapping par défaut)"
)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent

ADAPTER_PATHS = {
    "sft": BASE_DIR / "models" / "unsloth_sft_lora_2026-04-24_12-36",
    # Flux Unsloth "base Qwen + LoRA" (evite les erreurs de chargement du modele merge).
    "dpo": BASE_DIR / "models" / "unsloth_dpo_lora_2026-07-10_12-45",
}

BASE_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
ADAPTER_PATH   = Path(args.adapter_path) if args.adapter_path else ADAPTER_PATHS[args.adapter]
OUTPUT_JSON    = BASE_DIR / "data" / f"dspy_optimized_triage_{args.adapter}.json"
OUTPUT_PREVIEW = BASE_DIR / "data" / f"dspy_prompt_preview_{args.adapter}.txt"

print(f"[Config] Profil    : {args.adapter}")
print(f"[Config] Modèle    : {ADAPTER_PATH}")

if not ADAPTER_PATH.exists():
    print(f"[ERREUR] Modèle introuvable : {ADAPTER_PATH}")
    sys.exit(1)

# ==========================================
# 2. CHARGEMENT DU MODELE LOCAL
# ==========================================


print(f"[Model] Chargement Qwen3-1.7B + LoRA ({args.adapter})...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_PATH),
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)
model.eval()

# Template ChatML
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)
tokenizer.chat_template = CHATML_TEMPLATE
tokenizer.pad_token = tokenizer.eos_token

print("[Model] Modele charge avec succes.")

# ==========================================
# 3. FONCTION D'INFERENCE LOCALE
# ==========================================
def infer(messages: list[dict], max_new_tokens: int = 256) -> str:
    """Execute une inference sur le modele local (256 tokens = parite prod)."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    # <|im_end|> ET <|endoftext|> en tokens d'arret (le modele peut emettre l'un ou l'autre).
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    stop_ids = list({tokenizer.eos_token_id, im_end_id, endoftext_id})

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Extraire le premier bloc JSON valide si le modele a genere du texte supplementaire
    brace = raw.find("{")
    if brace != -1:
        for end in range(len(raw), brace, -1):
            candidate = raw[brace:end]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                continue
    return raw


# ==========================================
# 4. SYSTEME DE PROMPT TRIAGE
# ==========================================
# System prompt EXACT du dataset SFT v2.0.0 (s'en ecarter degrade le conditionnement) + regle 4 ajoutee pour le rappel Haute.
# Accolades simples : injecte via .replace(), pas .format().
SYSTEM_PROMPT_TEMPLATE = """\
Tu es un médecin urgentiste chargé de trier des situations cliniques.
Ton objectif est de décider entre deux actions :
- POSER UNE QUESTION si les informations sont insuffisantes ou ambiguës
- DONNER UN VERDICT MÉDICAL STRUCTURÉ si les informations sont suffisantes
Règles :
1. Tu dois toujours répondre au format JSON strict.
2. Si les informations sont insuffisantes, pose UNE seule question ciblée.
3. Si les informations sont suffisantes, donne une analyse médicale avec un niveau d'urgence.
4. Pour les urgences manifestes (signes FAST d'AVC, douleur thoracique aiguë avec sueurs ou dyspnée, perte de connaissance), donne DIRECTEMENT le verdict Haute sans poser de question.
5. L'urgence doit être strictement : "Haute", "Moyenne" ou "Faible".
6. Ne jamais inclure de texte hors JSON.
7. Sois concis, médicalement prudent et factuel.
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
}
{demos_block}"""


def build_system_with_demos(demos: list[dict]) -> str:
    """Construit le system prompt avec les exemples few-shot bootstrappes."""
    if not demos:
        return SYSTEM_PROMPT_TEMPLATE.replace("{demos_block}", "").strip()

    lines = ["\nExemples :"]
    for i, d in enumerate(demos, 1):
        lines.append(f"\nExemple {i} :")
        lines.append(f"Patient   : {d['symptomes']}")
        lines.append(f"Reponse   : {d['reponse']}")

    return SYSTEM_PROMPT_TEMPLATE.replace("{demos_block}", "\n".join(lines)).strip()


# ==========================================
# 5. DATASET GOLD (exemples d'entrainement)
# ==========================================
print("[Data] Preparation du dataset gold (12 exemples)...")

gold_examples = [
    # --- FRENCH: FINAL / ANALYSE ---
    {"symptomes": "Douleur violente dans la poitrine, du mal à respirer et je transpire beaucoup.",
     "reponse": '{"type": "final", "question": null, "urgence": "Haute", "analyse": "Signes évocateurs de syndrome coronaire aigu. Urgence absolue (SAMU/15)."}'},
     
    {"symptomes": "Je me suis coupé avec une feuille de papier, ça saigne très peu.",
     "reponse": '{"type": "final", "question": null, "urgence": "Faible", "analyse": "Plaie superficielle sans signe de gravité. Nettoyer et désinfecter."}'},

    # --- ENGLISH: FINAL / ANALYSE ---
    {"symptomes": "My 3-year-old child has a fever of 40 degrees Celsius and is having a seizure.",
     "reponse": '{"type": "final", "question": null, "urgence": "Haute", "analyse": "Febrile seizure in a child. Medical emergency. Call emergency services immediately."}'},
     
    {"symptomes": "I have a mild runny nose and a slight sore throat, no fever.",
     "reponse": '{"type": "final", "question": null, "urgence": "Faible", "analyse": "Mild viral infection (common cold). Rest and hydration. Consult if fever develops."}'},

    # --- FRENCH: QUESTION ---
    {"symptomes": "J'ai mal au ventre depuis ce matin.",
     "reponse": '{"type": "question", "question": "La douleur est-elle localisée d\'un côté précis et avez-vous de la fièvre ou des nausées ?", "urgence": null, "analyse": null}'},
     
    {"symptomes": "J'ai des vertiges quand je me lève.",
     "reponse": '{"type": "question", "question": "Est-ce que cela s\'accompagne d\'une perte d\'équilibre, de sifflements d\'oreilles ou de maux de tête ?", "urgence": null, "analyse": null}'},

    # --- ENGLISH: QUESTION ---
    {"symptomes": "I've been coughing a lot for the past two days.",
     "reponse": '{"type": "question", "question": "Is your cough dry or producing mucus, and are you experiencing any shortness of breath?", "urgence": null, "analyse": null}'},
     
    {"symptomes": "I have red, itchy patches on my arms.",
     "reponse": '{"type": "question", "question": "Have you eaten any new foods or used new products recently, and do you feel any swelling in your face or throat?", "urgence": null, "analyse": null}'},
     
    # Extra examples for validation (val_examples)
    # FRENCH FINAL
    {"symptomes": "J'ai de la fièvre à 38.8 depuis 3 jours avec des frissons et des douleurs en urinant.",
     "reponse": '{"type": "final", "question": null, "urgence": "Moyenne", "analyse": "Suspicion de pyélonéphrite ou infection urinaire basse. Consultation médicale recommandée dans la journée."}'},
    # ENGLISH FINAL
    {"symptomes": "I twisted my ankle playing soccer, it's very swollen and I can't put any weight on it.",
     "reponse": '{"type": "final", "question": null, "urgence": "Moyenne", "analyse": "Possible severe sprain or fracture. Apply ice, immobilize, and get an X-ray within 12 hours."}'},
    # FRENCH QUESTION
    {"symptomes": "Une femme de 32 ans consulte pour des douleurs abdominales intenses depuis 2 heures, associées à des nausées. Règles irrégulières.",
     "reponse": '{"type": "question", "question": "Avez-vous eu des rapports sexuels non protégés ou un retard de règles ces dernières semaines ?", "urgence": null, "analyse": null}'},
    # ENGLISH QUESTION
    {"symptomes": "A 72-year-old woman is brought in for sudden confusion over the past 2 hours, no fever.",
     "reponse": '{"type": "question", "question": "Have you noticed any recent signs of dehydration or has she started taking any new medications?", "urgence": null, "analyse": null}'},
]

split = int(len(gold_examples) * 0.8)
train_examples = gold_examples[:split]
val_examples   = gold_examples[split:]
print(f"[Data] Train : {len(train_examples)} | Val : {len(val_examples)}")


# ==========================================
# 6. METRIQUE D'EVALUATION
# ==========================================
def triage_metric(reponse: str) -> bool:
    """Verifie que la reponse est un JSON valide au format SFT triage."""
    reponse = (reponse or "").strip()
    # Nettoyer les balises markdown si presentes
    if reponse.startswith("```"):
        reponse = reponse.strip("`").strip()
        if reponse.startswith("json"):
            reponse = reponse[4:].strip()
    try:
        data = json.loads(reponse)
        t = data.get("type")
        if t == "final":
            return (
                data.get("urgence") in {"Haute", "Moyenne", "Faible"}
                and bool(data.get("analyse"))
                and data.get("question") is None
            )
        elif t == "question":
            return bool(data.get("question")) and data.get("urgence") is None
    except Exception:
        pass
    return False


# ==========================================
# 7. BOOTSTRAP FEW-SHOT MANUEL
# ==========================================
print("[Bootstrap] Selection des meilleures demos...")

# Quotas par categorie : sans demo final/Haute dans le prompt, le modele
# sur-questionne les urgences manifestes (rappel Haute 0.40 mesure le
# 10/07/2026 avec un prompt 3 question + 1 final/Faible ; seuil CI 0.90).
DEMO_QUOTAS = {"final_haute": 2, "final_autre": 1, "question": 2}


def demo_category(reponse_gold: str) -> str:
    gold = json.loads(reponse_gold)
    if gold.get("type") == "question":
        return "question"
    return "final_haute" if gold.get("urgence") == "Haute" else "final_autre"


demos_par_cat = {cat: [] for cat in DEMO_QUOTAS}

for ex in train_examples:
    cat = demo_category(ex["reponse"])
    if len(demos_par_cat[cat]) >= DEMO_QUOTAS[cat]:
        continue

    # Tester sur le modele sans demos d'abord
    system_base = build_system_with_demos([])
    messages = [
        {"role": "system", "content": system_base},
        {"role": "user",   "content": ex['symptomes']},
    ]

    pred = infer(messages, max_new_tokens=256)
    # Demo valide = JSON au bon format, de longueur raisonnable (800 chars ~=
    # p50 des reponses "final" du dataset, une demo verbeuse apprend la
    # verbosite par mimetisme) ET CONFORME AU GOLD : sans ce dernier critere,
    # une erreur de triage du modele (ex. final/Haute sur un cas vague)
    # devient une demo et biaise tout le prompt vers le sur-triage.
    format_ok = triage_metric(pred) and len(pred) <= 800
    gold_ok = False
    if format_ok:
        try:
            gold = json.loads(ex["reponse"])
            pred_data = json.loads(pred.strip().strip("`").removeprefix("json").strip())
            gold_ok = pred_data.get("type") == gold.get("type") and (
                gold.get("type") != "final"
                or pred_data.get("urgence") == gold.get("urgence")
            )
        except Exception:
            gold_ok = False
    ok = format_ok and gold_ok

    status = "OK" if ok else ("NON CONFORME GOLD" if format_ok else "ECHEC")
    print(f"  [{status}] [{cat}] {ex['symptomes'][:60]}...")

    if ok:
        # Le modele repond correctement -> on ajoute comme demo
        demos_par_cat[cat].append({
            "symptomes": ex["symptomes"],
            "reponse"  : pred,
        })

# Complement par categorie avec les reponses gold : le bootstrap echoue
# precisement sur les categories ou le modele est faible (ex. final/Haute),
# qui sont celles dont le prompt a le plus besoin.
for cat, quota in DEMO_QUOTAS.items():
    for ex in train_examples:
        if len(demos_par_cat[cat]) >= quota:
            break
        if demo_category(ex["reponse"]) != cat:
            continue
        deja = any(
            d["symptomes"] == ex["symptomes"]
            for demos in demos_par_cat.values() for d in demos
        )
        if not deja:
            print(f"  [GOLD] [{cat}] {ex['symptomes'][:60]}...")
            demos_par_cat[cat].append({
                "symptomes": ex["symptomes"],
                "reponse"  : ex["reponse"],
            })

# Haute d'abord : primaute des urgences manifestes dans le prompt
bootstrapped_demos = (
    demos_par_cat["final_haute"]
    + demos_par_cat["final_autre"]
    + demos_par_cat["question"]
)

print("[Bootstrap] " + " | ".join(
    f"{cat}: {len(v)}/{DEMO_QUOTAS[cat]}" for cat, v in demos_par_cat.items()
))

# ==========================================
# 8. EVALUATION VAL SET
# ==========================================
print("[Eval] Evaluation sur le val set avec les demos bootstrappees...")

system_with_demos = build_system_with_demos(bootstrapped_demos)
correct = 0
predictions = []

for ex in val_examples:
    messages = [
        {"role": "system", "content": system_with_demos},
        {"role": "user",   "content": ex['symptomes']},
    ]
    pred = infer(messages, max_new_tokens=256)
    # format + conformite au gold (type, et urgence pour les cas final)
    ok = triage_metric(pred)
    if ok:
        try:
            gold = json.loads(ex["reponse"])
            pred_data = json.loads(pred.strip().strip("`").removeprefix("json").strip())
            ok = pred_data.get("type") == gold.get("type") and (
                gold.get("type") != "final"
                or pred_data.get("urgence") == gold.get("urgence")
            )
        except Exception:
            ok = False
    correct += int(ok)

    predictions.append({
        "symptomes": ex["symptomes"],
        "gold":      ex["reponse"],
        "pred":      pred,
        "ok":        ok,
    })

accuracy = correct / len(val_examples) if val_examples else 0
print(f"[Eval] Accuracy : {correct}/{len(val_examples)} = {accuracy:.0%}")


# ==========================================
# 9. SAUVEGARDE JSON DU PROGRAMME COMPILE
# ==========================================
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

programme = {
    "adapter":          args.adapter,
    "base_model":       BASE_MODEL_ID,
    "model_path":       str(ADAPTER_PATH),
    "val_accuracy":     round(accuracy, 3),
    "n_demos":          len(bootstrapped_demos),
    "system_prompt":    system_with_demos,
    "demos":            bootstrapped_demos,
}

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(programme, f, ensure_ascii=False, indent=2)

print(f"\n[Save] Programme sauvegarde : {OUTPUT_JSON}")


# ==========================================
# 10. DISPLAY 
# ==========================================
lines = []
lines.append("=" * 70)
lines.append(f"DSPY TRIAGE MEDICAL - PROMPTS OPTIMISES ({args.adapter.upper()})")
lines.append(f"Modele  : Qwen3-1.7B-Base + LoRA ({args.adapter})")
lines.append(f"Accuracy val : {accuracy:.0%}  ({correct}/{len(val_examples)})")
lines.append(f"Demos bootstrappees : {len(bootstrapped_demos)}")
lines.append("=" * 70)

lines.append("\n--- SYSTEM PROMPT COMPILE ---")
lines.append(system_with_demos)

lines.append("\n--- DEMOS BOOTSTRAPPEES ---")
for i, d in enumerate(bootstrapped_demos, 1):
    lines.append(f"\n[Demo {i}]")
    lines.append(f"  Symptomes : {d['symptomes']}")
    lines.append(f"  Reponse   : {d['reponse']}")

lines.append("\n--- PREDICTIONS VAL SET ---")
for i, p in enumerate(predictions, 1):
    lines.append(f"\n[{i}] {'OK' if p['ok'] else 'ECHEC'}")
    lines.append(f"  Symptomes : {p['symptomes']}")
    lines.append(f"  Gold      : {p['gold']}")
    lines.append(f"  Pred      : {p['pred']}")

OUTPUT_PREVIEW.write_text("\n".join(lines), encoding="utf-8")
print(f"[Preview] Apercu sauvegarde : {OUTPUT_PREVIEW}")

print("\n[Done] Optimisation terminee.")
print(f"  -> JSON   : {OUTPUT_JSON}")
print(f"  -> Apercu : {OUTPUT_PREVIEW}")
