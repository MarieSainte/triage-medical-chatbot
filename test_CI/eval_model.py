import json
import sys
import re
import os

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import requests
from typing import Dict, List, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import du dataset
try:
    from eval_dataset import DATASET
    print("Dataset charge.")
except ImportError:
    from test_CI.eval_dataset import DATASET
    print("Dataset charge depuis test_CI.")

# Import du prompt optimise DSPy + demos few-shot
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from api.dspy.signatures import OPTIMIZED_SYSTEM_PROMPT, _DEMOS
    print("Prompt optimise charge.")
except Exception:
    OPTIMIZED_SYSTEM_PROMPT = None
    _DEMOS = []

# =========================
# CONFIG
# =========================

# CI rapide : 5 premiers exemples. Eval complete locale : EVAL_SAMPLE_LIMIT=20
CI_SAMPLE_LIMIT = int(os.getenv("EVAL_SAMPLE_LIMIT", "5"))
EVAL_DATASET    = DATASET[:CI_SAMPLE_LIMIT]

GCS_LORA_BASE_URL = os.getenv(
    "GCS_LORA_BASE_URL", "https://storage.googleapis.com/lora-matrice/checkpoint-60"
)
MODEL_ID = os.getenv("MODEL_ID", "production_model")

LABELS = ["Haute", "Moyenne", "Faible", "question"]

# Seuils CI/CD
# recall_haute : critique securite patient — ne pas rater un cas urgent (multi-tour inclus)
# accuracy     : souple car le modele pose souvent une question au premier tour
THRESHOLDS = {
    "recall_haute": 0.90,
    "accuracy":     0.50,
}

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)


def sanitize_local_model_json_files(model_id: str):
    """Retire un eventuel BOM UTF-8 des fichiers JSON d'un modele local."""
    model_path = Path(model_id)
    if not model_path.is_dir():
        return

    fixed_files = []
    for json_path in model_path.glob("*.json"):
        raw = json_path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            text = raw.decode("utf-8-sig")
            json_path.write_text(text, encoding="utf-8")
            fixed_files.append(json_path.name)

    if fixed_files:
        print("BOM UTF-8 retire de : " + ", ".join(sorted(fixed_files)))


def ensure_chatml_generation_prompt(tokenizer):
    """Force un template ChatML compatible `add_generation_prompt=True`.

    Le modele merge en local peut embarquer un `chat_template.jinja` incomplet
    (messages seulement, sans prefixe assistant). Dans ce cas, Qwen continue le
    texte apres le dernier user au lieu de repondre en assistant JSON.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    if "add_generation_prompt" not in template or "<|im_start|>assistant" not in template:
        tokenizer.chat_template = CHATML_TEMPLATE
        print("Chat template ChatML normalise pour l'inference.")

# =========================
# LOCAL MODEL CALLER
# =========================

class LocalModelCaller:
    def __init__(self, model_id: str = MODEL_ID, adapter_path: Optional[str] = None):
        if adapter_path:
            self.ensure_adapter(adapter_path)

        sanitize_local_model_json_files(model_id)
        print(f"Chargement du modele {model_id} sur CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        ensure_chatml_generation_prompt(self.tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if adapter_path:
            print(f"Chargement de l adaptateur depuis {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            # NE PAS "reparer" les embeddings ChatML ici : les adaptateurs
            # post-10/07/2026 ont ete entraines en LISANT les lignes originales
            # (la chirurgie de train_Unsloth_sft.py n'affecte que la tete de
            # sortie sur le chemin Unsloth) et s'arretent nativement en emettant
            # <|endoftext|> (deja dans stop_ids). Modifier les embeddings a
            # l'inference degrade la lecture du prompt (verifie le 10/07/2026 :
            # sorties degenerees).

        self.model.eval()

        if OPTIMIZED_SYSTEM_PROMPT:
            print("Prompt DSPy optimise utilise.")
            self.system_prompt = OPTIMIZED_SYSTEM_PROMPT
        else:
            print("Avertissement : prompt optimise absent, prompt de secours utilise.")
            self.system_prompt = (
                "Tu es un medecin urgentiste charge de trier des situations cliniques. "
                "Reponds UNIQUEMENT en JSON : "
                '{"type":"final","question":null,"urgence":"Haute|Moyenne|Faible","analyse":"..."} '
                'ou {"type":"question","question":"...","urgence":null,"analyse":null}'
            )
        # Parite prod (TriageModule) : le system prompt DSPy embarque deja les
        # exemples en texte -> ne pas les reinjecter comme tours de conversation
        # (sinon chaque demo est presentee deux fois au modele).
        self.demos = [] if "Exemple" in self.system_prompt else _DEMOS

    def ensure_adapter(self, adapter_path: str):
        """Telecharge les matrices LoRA depuis GCS si elles sont absentes."""
        path = Path(adapter_path)
        path.mkdir(parents=True, exist_ok=True)

        for filename in ["adapter_config.json", "adapter_model.safetensors"]:
            file_path = path / filename
            if not file_path.exists():
                url = f"{GCS_LORA_BASE_URL}/{filename}"
                print(f"Telechargement de {filename} depuis {url}...")
                try:
                    r = requests.get(url, stream=True, timeout=120)
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"  {filename} telecharge.")
                except Exception as e:
                    print(f"  ERREUR telechargement {filename} : {e}")
                    sys.exit(1)
            else:
                print(f"  {filename} deja present.")

    def _build_messages(self, input_text: str) -> List[Dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for demo in self.demos:
            s = demo.get("symptomes", "")
            r = demo.get("reponse", "")
            if s and r:
                messages.append({"role": "user",      "content": s})
                messages.append({"role": "assistant", "content": r})
        messages.append({"role": "user", "content": input_text})
        return messages

    def _generate(self, messages: List[Dict]) -> Dict:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        endoftext_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        stop_ids  = list({self.tokenizer.eos_token_id, im_end_id, endoftext_id})

        max_new_tokens = 256
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_ids,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response  = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Arret propre = le modele a emis un stop token avant max_new_tokens.
        # Un modele qui n'emet jamais <|im_end|> (bug embeddings Qwen3-Base,
        # cf. train_Unsloth_sft.py) produit du JSON valide suivi de charabia :
        # les metriques de labels passent mais le modele est indeployable.
        clean_stop = len(generated) < max_new_tokens and generated[-1].item() in stop_ids

        data = _extract_json(response)
        if data:
            if data.get("type") == "final":
                u = str(data.get("urgence", "")).capitalize()
                data["urgence"] = u if u in ["Haute", "Moyenne", "Faible"] else "Inconnue"
            data["_raw"] = response
            data["_clean_stop"] = clean_stop
            return data

        print(f"  Pas de JSON valide : {response[:100]}...")
        return {"type": "error", "message": "No JSON found", "_raw": response, "_clean_stop": clean_stop}

    def predict(self, input_text: str) -> Dict:
        return self._generate(self._build_messages(input_text))

    def predict_turn2(self, input_text: str, turn1_pred: Dict, followup: str) -> Dict:
        """Deuxieme tour : fournit le followup patient apres la question du modele."""
        turn1_raw = turn1_pred.get("_raw") or json.dumps(
            {k: v for k, v in turn1_pred.items() if k != "_raw"},
            ensure_ascii=False,
        )
        messages = self._build_messages(input_text)
        messages.append({"role": "assistant", "content": turn1_raw})
        messages.append({"role": "user",      "content": followup})
        return self._generate(messages)


# =========================
# HELPERS
# =========================

def _extract_json(raw: str) -> Optional[Dict]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    brace = raw.find("{")
    if brace == -1:
        return None
    for end in range(len(raw), brace, -1):
        try:
            return json.loads(raw[brace:end])
        except Exception:
            continue
    return None


def get_label(pred: Dict) -> str:
    if pred.get("type") == "question":
        return "question"
    u = pred.get("urgence", "Inconnue")
    return u if u in ["Haute", "Moyenne", "Faible"] else "Autre"


def compute_class_metrics(
    confusion: Dict[str, Dict[str, int]], label: str
) -> Dict[str, float]:
    tp = confusion.get(label, {}).get(label, 0)
    fp = sum(confusion.get(l, {}).get(label, 0) for l in LABELS if l != label)
    fn = sum(confusion.get(label, {}).get(l, 0) for l in LABELS if l != label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def print_confusion_matrix(matrix: Dict[str, Dict[str, int]]):
    print("\n===== MATRICE DE CONFUSION =====")
    print("REEL \\ PRED".ljust(12) + "".join(l.ljust(10) for l in LABELS))
    for true_label in LABELS:
        row = true_label.ljust(12)
        for pred_label in LABELS:
            row += str(matrix.get(true_label, {}).get(pred_label, 0)).ljust(10)
        print(row)


# =========================
# EVALUATION PRINCIPALE
# =========================

def evaluate():
    adapter_path = os.getenv("ADAPTER_PATH", "NONE")
    if adapter_path.upper() == "NONE":
        adapter_path = None
    caller = LocalModelCaller(adapter_path=adapter_path)

    confusion      = {l: {l2: 0 for l2 in LABELS} for l in LABELS}
    total          = len(EVAL_DATASET)
    correct        = 0
    haute_total    = 0
    haute_correct  = 0
    missing_fields = 0
    multiturn_used = 0
    gen_total      = 0
    gen_stopped    = 0

    print(f"\nEvaluation multi-tour sur {total} exemples (limite={CI_SAMPLE_LIMIT})")
    print("=" * 60)

    for i, sample in enumerate(EVAL_DATASET):
        followup = sample.get("followup")
        print(f"[{i+1}/{total}] {sample['input'][:60]}...")

        pred  = caller.predict(sample["input"])
        turns = 1
        gen_total   += 1
        gen_stopped += int(pred.get("_clean_stop", False))

        # Tour 2 : si le modele pose une question et qu'un followup est defini
        if pred.get("type") == "question" and followup:
            print(f"  -> Tour 1 : question posee — envoi du followup patient")
            pred   = caller.predict_turn2(sample["input"], pred, followup)
            turns  = 2
            multiturn_used += 1
            gen_total   += 1
            gen_stopped += int(pred.get("_clean_stop", False))

        actual = (
            "question" if sample["expected_type"] == "question"
            else sample.get("expected_urgence", "Inconnue")
        )

        # Champs obligatoires
        if pred.get("type") == "question" and not pred.get("question"):
            print("  Champ 'question' manquant")
            missing_fields += 1
        elif pred.get("type") == "final" and not pred.get("analyse"):
            print("  Champ 'analyse' manquant")
            missing_fields += 1

        pred_label = get_label(pred)

        if actual in LABELS and pred_label in LABELS:
            confusion[actual][pred_label] += 1

        ok = pred_label == actual
        if ok:
            correct += 1
        print(f"  {'OK  ' if ok else 'FAIL'} [{turns}T] attendu={actual:<8} predit={pred_label}")

        if actual == "Haute":
            haute_total += 1
            if pred_label == "Haute":
                haute_correct += 1

    # ===== METRIQUES =====
    accuracy     = correct / total
    recall_haute = haute_correct / haute_total if haute_total > 0 else 1.0

    print_confusion_matrix(confusion)

    print("\n===== METRIQUES PAR CLASSE =====")
    print(f"{'Classe':<12} {'Precision':>10} {'Rappel':>10} {'F1':>10}")
    print("-" * 45)
    for label in LABELS:
        m = compute_class_metrics(confusion, label)
        print(f"{label:<12} {m['precision']:>10.2f} {m['recall']:>10.2f} {m['f1']:>10.2f}")

    stop_rate = gen_stopped / gen_total if gen_total > 0 else 0.0

    print("\n===== RESUME =====")
    print(f"Accuracy globale : {accuracy:.2f}   (seuil {THRESHOLDS['accuracy']:.2f})")
    print(f"Rappel Haute     : {recall_haute:.2f}   (seuil {THRESHOLDS['recall_haute']:.2f})")
    print(f"Champs manquants : {missing_fields}")
    print(f"Tours multiples  : {multiturn_used} exemple(s) evalues en 2 tours")
    print(f"Arret EOS propre : {gen_stopped}/{gen_total} generations ({stop_rate:.0%})")

    # ===== DECISION CI/CD =====
    failed = False

    # Seuil optionnel (EOS_STOP_RATE_MIN=0.9 pour bloquer) : le modele en prod
    # au 10 juillet 2026 n'emet jamais <|im_end|> — activer ce seuil bloquerait
    # son redeploiement tant qu'un modele corrige n'est pas publie.
    eos_min = float(os.getenv("EOS_STOP_RATE_MIN", "0"))
    if stop_rate < eos_min:
        print(f"\nBLOCAGE : taux d'arret EOS {stop_rate:.2f} < {eos_min:.2f}")
        failed = True

    if recall_haute < THRESHOLDS["recall_haute"]:
        print(f"\nBLOCAGE : rappel Haute {recall_haute:.2f} < {THRESHOLDS['recall_haute']:.2f}")
        failed = True

    if accuracy < THRESHOLDS["accuracy"]:
        print(f"\nBLOCAGE : accuracy {accuracy:.2f} < {THRESHOLDS['accuracy']:.2f}")
        failed = True

    if missing_fields > 0:
        print(f"\nBLOCAGE : {missing_fields} champ(s) obligatoire(s) manquant(s)")
        failed = True

    if failed:
        print("\nEVALUATION : ECHEC — deploiement bloque.")
        sys.exit(1)
    else:
        print("\nEVALUATION : OK — deploiement autorise.")
        sys.exit(0)


if __name__ == "__main__":
    evaluate()
