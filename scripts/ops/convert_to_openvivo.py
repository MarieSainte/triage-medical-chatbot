import torch
import subprocess
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURATION DES CHEMINS ---
base_model_id = "Qwen/Qwen3-1.7B-Base" 
lora_weights_path = Path("models/unsloth_dpo_lora_2026-04-15_03-21/checkpoint-60")

# Dossiers de sortie
export_dir = Path("models/merged/qwen-cpu-merged")
openvino_dir = Path("models/qwen-cpu-openvino-int4")

print(f"🚀 Chargement du modèle de base : {base_model_id}")
# 1. Charger le tokenizer et le modèle de base en FP16
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu" # On merge sur CPU pour éviter les soucis de VRAM
)
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

# Appliquer le template au tokenizer
tokenizer.chat_template = CHATML_TEMPLATE
print(f"🧩 Chargement des adaptateurs LoRA depuis : {lora_weights_path}")
# 2. Charger les matrices LoRA par-dessus
model = PeftModel.from_pretrained(base_model, lora_weights_path)

print("🔄 Fusion des poids en cours...")
# 3. Merger les matrices (Merge & Unload)
merged_model = model.merge_and_unload()

# S'assurer que le dossier parent existe
export_dir.mkdir(parents=True, exist_ok=True)

print(f"💾 Sauvegarde du modèle complet dans : {export_dir}")
# 4. Sauvegarder le modèle complet ET le tokenizer (toujours au même endroit !)
merged_model.save_pretrained(export_dir)
tokenizer.save_pretrained(export_dir)

# 5. Préparation et lancement de la commande OpenVINO
print("\n⚡ Démarrage de la conversion OpenVINO...")

# On utilise les variables Path directement dans la chaîne pour éviter les erreurs de frappe
cli_cmd = f"uv run optimum-cli export openvino --model {export_dir} --task text-generation --weight-format int4 {openvino_dir}"

print(f"Exécution : {cli_cmd}")

try:
    # check=True fait planter le script Python si la commande OpenVINO échoue
    subprocess.run(cli_cmd, shell=True, check=True)
    print(f"\n✅ Conversion terminée avec succès ! Modèle prêt dans : {openvino_dir}")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Erreur lors de la conversion OpenVINO : {e}")