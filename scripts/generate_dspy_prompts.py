
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

# ==========================================
# 1. ARGUMENTS CLI
# ==========================================
parser = argparse.ArgumentParser(description="Optimisation DSPy des prompts triage")
parser.add_argument(
    "--adapter",
    choices=["sft", "dpo"],
    default="sft",
    help="Adapteur LoRA : 'sft' (modele_final_lora) ou 'dpo' (modele_final_DPO/modele_dpo_lora)"
)
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent

ADAPTER_PATHS = {
    "sft": BASE_DIR / "models" / "unsloth_sft_lora_2026-04-14_16-11",
    "dpo": BASE_DIR / "models" / "unsloth_dpo_lora_2026-04-15_03-21" / "checkpoint-60",
}

BASE_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
ADAPTER_PATH   = ADAPTER_PATHS[args.adapter]
OUTPUT_JSON    = BASE_DIR / "data" / f"dspy_optimized_triage_{args.adapter}.json"
OUTPUT_PREVIEW = BASE_DIR / "data" / f"dspy_prompt_preview_{args.adapter}.txt"

print(f"[Config] Adapteur  : {args.adapter}")
print(f"[Config] Chemin    : {ADAPTER_PATH}")

if not ADAPTER_PATH.exists():
    print(f"[ERREUR] Adapteur introuvable : {ADAPTER_PATH}")
    sys.exit(1)

# ==========================================
# 2. CHARGEMENT DU MODELE LOCAL
# ==========================================
import unsloth
from unsloth import FastLanguageModel
import torch

print(f"[Model] Chargement Qwen3-1.7B + LoRA ({args.adapter})...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_PATH),
    max_seq_length=1024,
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
def infer(messages: list[dict], max_new_tokens: int = 300) -> str:
    """Execute une inference sur le modele local."""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=1024
    ).to(model.device)

    # Inclure <|im_end|> comme token d'arret 
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [tokenizer.eos_token_id, im_end_id]

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
SYSTEM_PROMPT_TEMPLATE = """\
Tu es un medecin urgentiste charge de trier des situations cliniques.
Ton objectif est de decider entre deux actions :
- POSER UNE QUESTION si les informations sont insuffisantes ou ambigues
- DONNER UN VERDICT MEDICAL STRUCTURE si les informations sont suffisantes
Regles :
1. Tu dois toujours repondre au format JSON strict.
2. Si les informations sont insuffisantes, pose UNE seule question ciblee.
3. Si les informations sont suffisantes, donne une analyse medicale avec un niveau d'urgence.
4. L'urgence doit etre strictement : "Haute", "Moyenne" ou "Faible".
5. Ne jamais inclure de texte hors JSON.
6. Sois concis, medicalement prudent et factuel.
Format attendu :
CAS QUESTION :
{{"type": "question", "question": "...", "urgence": null, "analyse": null}}
CAS FINAL :
{{"type": "final", "question": null, "urgence": "...", "analyse": "..."}}
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
    # URGENCE HAUTE
    {"symptomes": "Douleur violente dans la poitrine, du mal a respirer et je transpire beaucoup.",
     "reponse": '{"type": "final", "question": null, "urgence": "Haute", "analyse": "Signes evocateurs de syndrome coronaire aigu. Appeler le 15 immediatement."}'},
    {"symptomes": "Je vois flou d'un seul coup de l'oeil droit et j'ai une partie du visage qui semble paralysee.",
     "reponse": '{"type": "final", "question": null, "urgence": "Haute", "analyse": "Signes suspects d\'AVC. Appeler le 15, noter l\'heure d\'apparition."}'},
    {"symptomes": "Ma jambe est toute rouge, gonflee et tres chaude apres une chirurgie la semaine derniere.",
     "reponse": '{"type": "final", "question": null, "urgence": "Haute", "analyse": "Suspicion thrombose veineuse profonde. Aller aux urgences immediatement."}'},
    {"symptomes": "Mon enfant de 3 ans a une fievre a 40 degres et fait des convulsions.",
     "reponse": '{"type": "final", "question": null, "urgence": "Haute", "analyse": "Convulsions febriles enfant. Appeler le 15 immediatement."}'},
    # URGENCE MOYENNE
    {"symptomes": "Je me suis tordu la cheville au foot, elle a double de volume et je ne peux plus poser le pied.",
     "reponse": '{"type": "final", "question": null, "urgence": "Moyenne", "analyse": "Suspicion entorse grave ou fracture. Glace, immobilisation, radio < 12h."}'},
    {"symptomes": "J'ai de la fievre depuis 3 jours a 38.8 avec des frissons et des douleurs en urinant.",
     "reponse": '{"type": "final", "question": null, "urgence": "Moyenne", "analyse": "Suspicion pyelonephrite. Consultation medicale dans la journee."}'},
    # URGENCE FAIBLE
    {"symptomes": "Je me suis coupe avec une feuille de papier, ca saigne tres peu mais ca pique.",
     "reponse": '{"type": "final", "question": null, "urgence": "Faible", "analyse": "Plaie superficielle sans signe de gravite. Nettoyer, desinfecter, pansement."}'},
    {"symptomes": "J'ai un gros rhume avec le nez qui coule et un peu mal a la gorge, pas de fievre.",
     "reponse": '{"type": "final", "question": null, "urgence": "Faible", "analyse": "Infection virale benigne. Lavage nez, repos. Consulter si fievre > 38.5."}'},
    # QUESTION
    {"symptomes": "J'ai mal au ventre depuis ce matin.",
     "reponse": '{"type": "question", "question": "La douleur est-elle localisee d\'un cote precis et avez-vous de la fievre ou des nausees ?", "urgence": null, "analyse": null}'},
    {"symptomes": "J'ai des vertiges quand je me leve.",
     "reponse": '{"type": "question", "question": "Est-ce que cela s\'accompagne d\'une perte d\'equilibre, de sifflements d\'oreilles ou de maux de tete ?", "urgence": null, "analyse": null}'},
    {"symptomes": "Je tousse beaucoup depuis deux jours.",
     "reponse": '{"type": "question", "question": "Votre toux est-elle grasse ou seche, et avez-vous des difficultes a reprendre votre souffle ?", "urgence": null, "analyse": null}'},
    {"symptomes": "J'ai des plaques rouges sur le bras qui grattent.",
     "reponse": '{"type": "question", "question": "Avez-vous mange un nouvel aliment ou utilise un nouveau produit, et ressentez-vous un gonflement du visage ?", "urgence": null, "analyse": null}'},
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

bootstrapped_demos = []
MAX_DEMOS = 4

for ex in train_examples:
    if len(bootstrapped_demos) >= MAX_DEMOS:
        break

    # Tester sur le modele sans demos d'abord
    system_base = build_system_with_demos([])
    messages = [
        {"role": "system", "content": system_base},
        {"role": "user",   "content": ex['symptomes']},
    ]

    pred = infer(messages, max_new_tokens=200)
    ok = triage_metric(pred)

    status = "OK" if ok else "ECHEC"
    print(f"  [{status}] {ex['symptomes'][:60]}...")

    if ok:
        # Le modele repond correctement -> on ajoute comme demo
        bootstrapped_demos.append({
            "symptomes": ex["symptomes"],
            "reponse"  : pred, 
        })

print(f"[Bootstrap] {len(bootstrapped_demos)} demos bootstrappees sur {len(train_examples)} exemples train.")

# Si on n'en a pas assez, on complète avec les gold examples
if len(bootstrapped_demos) < 2:
    print("[Bootstrap] Complement avec les exemples gold...")
    for ex in train_examples:
        if len(bootstrapped_demos) >= MAX_DEMOS:
            break
        already = any(d["symptomes"] == ex["symptomes"] for d in bootstrapped_demos)
        if not already:
            bootstrapped_demos.append({
                "symptomes": ex["symptomes"],
                "reponse"  : ex["reponse"],
            })

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
    pred = infer(messages, max_new_tokens=200)
    ok = triage_metric(pred)
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
    "adapter_path":     str(ADAPTER_PATH),
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
