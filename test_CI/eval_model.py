import requests
import json
import sys
import time
from typing import List, Dict
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import du dataset
try:
    from eval_dataset import DATASET
    print("Dataset chargé.")
except ImportError:
    from test_CI.eval_dataset import DATASET
    print("Dataset chargé depuis test_CI.")

# Import du prompt optimisé DSPy
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from api.dspy.signatures import OPTIMIZED_SYSTEM_PROMPT
    print("Prompt optimisé chargé.")
except Exception:
    OPTIMIZED_SYSTEM_PROMPT = None


# =========================
# CONFIG
# =========================

TEST_MODE = os.getenv("TEST_MODE", "API") 
API_URL = "http://localhost:8000/triage/ask"  
HEALTH_URL = "http://localhost:8000/health"
GCS_LORA_BASE_URL = os.getenv("GCS_LORA_BASE_URL", "https://storage.googleapis.com/lora-matrice")

# seuils CI/CD
THRESHOLDS = {
    "recall_haute": 0.80,
    "accuracy": 0.60
}

class LocalModelCaller:
    def __init__(self, model_id="Qwen/Qwen3-1.7B-Base", adapter_path=None):
        if adapter_path:
            self.ensure_adapter(adapter_path)

        print(f"Loading model {model_id} on CPU...")
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
            print(f"Loading adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        self.model.eval()
        
        if OPTIMIZED_SYSTEM_PROMPT:
            print(" Using optimized DSPy system prompt.")
            self.system_prompt = OPTIMIZED_SYSTEM_PROMPT
        else:
            print(" Warning: optimized prompt not found, using fallback.")
            self.system_prompt = """Tu es un médecin urgentiste chargé de trier des situations cliniques.
Ton objectif est de décider entre deux actions :
- POSER UNE QUESTION si les informations sont insuffisantes ou ambiguës
- DONNER UN VERDICT MÉDICAL STRUCTURÉ si les informations sont suffisantes
Règles :
1. Tu dois toujours répondre au format JSON strict.
2. Si les informations sont insuffisantes, pose UNE seule question ciblée.
3. Si les informations sont suffisantes, donne une analyse médicale avec un niveau d'urgence.
4. L'urgence doit être strictement : "Haute", "Moyenne" ou "Faible".
5. Ne jamais inclure de texte hors JSON.
6. Sois concis, médicalement prudent et factuel.
Format attendu :
{
  "type": "question",
  "question": "...",
  "urgence": null,
  "analyse": null
}
ou
{
  "type": "final",
  "question": null,
  "urgence": "...",
  "analyse": "..."
}"""

    def ensure_adapter(self, adapter_path: str):
        """Telecharge les matrices LoRA depuis GCS si elles sont absentes."""
        path = Path(adapter_path)
        path.mkdir(parents=True, exist_ok=True)
        
        files = ["adapter_config.json", "adapter_model.safetensors"]
        
        for filename in files:
            file_path = path / filename
            if not file_path.exists():
                url = f"{GCS_LORA_BASE_URL}/{filename}"
                print(f"📥 Téléchargement de {filename} depuis {url}...")
                try:
                    r = requests.get(url, stream=True, timeout=60)
                    r.raise_for_status()
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f" {filename} téléchargé.")
                except Exception as e:
                    print(f" Erreur lors du téléchargement de {filename} : {e}")
            else:
                print(f" {filename} déjà présent localement.")

    def predict(self, input_text: str) -> Dict:
        import torch
        prompt = f"{self.system_prompt}\n\n### USER\n{input_text}\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_output[len(prompt):].strip()

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > 0:
                data = json.loads(response[start:end])
                if data.get("type") == "final":
                    u = str(data.get("urgence")).capitalize()
                    data["urgence"] = u if u in ["Haute", "Moyenne", "Faible"] else "Inconnue"
                return data
            else:
                print(f" No JSON found in response: {response[:100]}...")
                return {"type": "error", "message": "No JSON found"}
        except Exception as e:
            print(f"Error parsing JSON: {e} | Response was: {response[:100]}...")
            return {"type": "error", "message": str(e)}

MODEL_CALLER = None

def get_model_caller():
    global MODEL_CALLER
    if MODEL_CALLER is None:
        default_adapter = "models/lora_triage"
        adapter = os.getenv("ADAPTER_PATH", default_adapter)
        MODEL_CALLER = LocalModelCaller(adapter_path=adapter)
    return MODEL_CALLER

# =========================
# CALL DISPATCHER
# =========================

def call_model(prompt: str) -> Dict:
    if TEST_MODE == "DIRECT":
        caller = get_model_caller()
        return caller.predict(prompt)
    else:
        response = requests.post(API_URL, json={"symptomes": prompt}, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "data" in data and data["data"]:
            return data["data"]
        return data

# =========================
# HELPERS
# =========================

def get_label(prediction: Dict) -> str:
    if prediction.get("type") == "question":
        return "question"
    
    urgence = prediction.get("urgence", "Inconnue")
    if urgence in ["Haute", "Moyenne", "Faible"]:
        return urgence
    
    return "Autre"

def print_confusion_matrix(matrix: Dict[str, Dict[str, int]], labels: List[str]):
    print("\n===== CONFUSION MATRIX =====")
    header = "TRUE \\ PRED".ljust(12) + "".join([l.ljust(10) for l in labels])
    print(header)
    for true_label in labels:
        row = true_label.ljust(12)
        for pred_label in labels:
            count = matrix.get(true_label, {}).get(pred_label, 0)
            row += str(count).ljust(10)
        print(row)

# =========================
# EVALUATION
# =========================

def evaluate():
    labels = ["Haute", "Moyenne", "Faible", "question"]
    confusion_matrix = {l: {l2: 0 for l2 in labels} for l in labels}
    
    total = len(DATASET)
    correct = 0
    haute_total = 0
    haute_correct = 0
    missing_fields = 0

    print(f"Starting evaluation on {total} samples (Mode: {TEST_MODE})...")

    for i, sample in enumerate(DATASET):
        print(f"[{i+1}/{total}] Processing: {sample['input'][:50]}...")
        pred = call_model(sample["input"])
        
        if sample["expected_type"] == "question":
            actual_label = "question"
            if not pred.get("question") and pred.get("type") == "question":
                print(f" Champ 'question' manquant")
                missing_fields += 1
        else:
            actual_label = sample.get("expected_urgence", "Inconnue")
            if not pred.get("analyse") and pred.get("type") == "final":
                print(f" Champ 'analyse' manquant")
                missing_fields += 1
            
        pred_label = get_label(pred)
        
        if actual_label in labels and pred_label in labels:
            confusion_matrix[actual_label][pred_label] += 1

        if pred_label == actual_label:
            correct += 1

        if actual_label == "Haute":
            haute_total += 1
            if pred_label == "Haute":
                haute_correct += 1

    accuracy = correct / total
    recall_haute = haute_correct / haute_total if haute_total > 0 else 1

    print("\n===== METRICS =====")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall Haute: {recall_haute:.2f}")
    print(f"Champs manquants: {missing_fields}")
    
    print_confusion_matrix(confusion_matrix, labels)

    failed = False
    if recall_haute < THRESHOLDS["recall_haute"]:
        print(f" FAIL: recall haute trop bas ({recall_haute:.2f} < {THRESHOLDS['recall_haute']:.2f})")
        failed = True
    if accuracy < THRESHOLDS["accuracy"]:
        print(f" FAIL: accuracy trop basse ({accuracy:.2f} < {THRESHOLDS['accuracy']:.2f})")
        failed = True
    if missing_fields > 0:
        print(f" FAIL: {missing_fields} champs obligatoires manquants")
        failed = True

    sys.exit(1 if failed else 0)

def wait_for_server(url: str, timeout: int = 30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("✅ Serveur prêt")
                return True
        except requests.ConnectionError:
            pass
        print("... attente du serveur")
        time.sleep(2)
    print(" Timeout serveur")
    return False

if __name__ == "__main__":
    if TEST_MODE == "DIRECT":
        evaluate()
    else:
        if wait_for_server(HEALTH_URL):
            evaluate()
        else:
            sys.exit(1)