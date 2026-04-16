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

# Import du prompt optimise DSPy
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from api.dspy.signatures import OPTIMIZED_SYSTEM_PROMPT
    print("Prompt optimise charge.")
except Exception:
    OPTIMIZED_SYSTEM_PROMPT = None

# =========================
# CONFIG
# =========================

# CI rapide : 5 premiers exemples. Eval complete locale : EVAL_SAMPLE_LIMIT=10
CI_SAMPLE_LIMIT = int(os.getenv("EVAL_SAMPLE_LIMIT", "5"))
EVAL_DATASET    = DATASET[:CI_SAMPLE_LIMIT]

GCS_LORA_BASE_URL = os.getenv(
    "GCS_LORA_BASE_URL", "https://storage.googleapis.com/lora-matrice/checkpoint-60"
)
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-1.7B-Base")

LABELS = ["Haute", "Moyenne", "Faible", "question"]

# Seuils CI/CD
# recall_haute : critique securite patient — ne pas rater un cas urgent
# accuracy     : souple car le modele est medicalement conservateur (sur-triage plutot qu'evitement)
THRESHOLDS = {
    "recall_haute": 0.90,
    "accuracy":     0.50,
}

# =========================
# LOCAL MODEL CALLER
# =========================

class LocalModelCaller:
    def __init__(self, model_id: str = MODEL_ID, adapter_path: Optional[str] = None):
        if adapter_path:
            self.ensure_adapter(adapter_path)

        print(f"Chargement du modele {model_id} sur CPU...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        if adapter_path:
            print(f"Chargement de l adaptateur depuis {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

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

    def predict(self, input_text: str) -> Dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": input_text},
        ]
        # Template natif du modele (gere add_generation_prompt correctement)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        stop_ids  = [self.tokenizer.eos_token_id, im_end_id]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_ids,
            )

        # Tokens generes uniquement (sans le prompt d entree)
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response  = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        data = _extract_json(response)
        if data:
            if data.get("type") == "final":
                u = str(data.get("urgence", "")).capitalize()
                data["urgence"] = u if u in ["Haute", "Moyenne", "Faible"] else "Inconnue"
            return data

        print(f"  Pas de JSON valide : {response[:100]}...")
        return {"type": "error", "message": "No JSON found"}


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
    adapter_path = os.getenv("ADAPTER_PATH", "models/lora_triage")
    caller = LocalModelCaller(adapter_path=adapter_path)

    confusion     = {l: {l2: 0 for l2 in LABELS} for l in LABELS}
    total         = len(EVAL_DATASET)
    correct       = 0
    haute_total   = 0
    haute_correct = 0
    missing_fields = 0

    print(f"\nEvaluation sur {total} exemples (limite={CI_SAMPLE_LIMIT})")
    print("=" * 60)

    for i, sample in enumerate(EVAL_DATASET):
        print(f"[{i+1}/{total}] {sample['input'][:60]}...")
        pred = caller.predict(sample["input"])

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
        print(f"  {'OK  ' if ok else 'FAIL'} attendu={actual:<8} predit={pred_label}")

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

    print("\n===== RESUME =====")
    print(f"Accuracy globale : {accuracy:.2f}   (seuil {THRESHOLDS['accuracy']:.2f})")
    print(f"Rappel Haute     : {recall_haute:.2f}   (seuil {THRESHOLDS['recall_haute']:.2f})")
    print(f"Champs manquants : {missing_fields}")

    # ===== DECISION CI/CD =====
    failed = False

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
