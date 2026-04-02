import pyarrow as pa
if not hasattr(pa, 'PyExtensionType'):
    pa.PyExtensionType = pa.ExtensionType
# --------------------------------------------------

import os
from pathlib import Path

# 1. On crée un dossier dédié sur le D:
cache_dir = Path(r"D:\hf_cache")

# 2. On redirige TOUS les flux de cache vers le D:
os.environ["HF_HOME"] = str(cache_dir)
os.environ["HF_HUB_CACHE"] = str(cache_dir)
os.environ["TMPDIR"] = str(cache_dir) # Pour la reconstruction des fichiers
os.environ["TEMP"] = str(cache_dir)   # Pour Windows
os.environ["TMP"] = str(cache_dir)    # Pour Windows

import torch
import mlflow
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from torch.cuda.amp import GradScaler, autocast

# ==========================================
# 1. PARAMÈTRES DU PROJET (MODE DRY-RUN)
# ==========================================
# Mettez MODE_TEST = True pour valider la pipeline sur 50 exemples en 2 minutes
# Mettez MODE_TEST = False pour lancer le vrai entraînement de nuit
MODE_TEST = False 
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR.parent / "medical-chatbot" / "data" / "data_versioned" / "sft"


model_id = "Qwen/Qwen3-1.7B-Base"
output_dir = "./models"
mlflow_experiment_name = "Triage_SFT_POC"

print(f"🔍 Vérification du chemin : {DATASET_PATH}")
# ==========================================
# 2. INITIALISATION MLFLOW
# ==========================================
print("📊 Initialisation de MLflow...")
# Crée un dossier local mlruns pour stocker vos graphiques et métriques
mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment(mlflow_experiment_name)

# ==========================================
# 3. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==========================================

train_file = DATASET_PATH / "sft_train_v1.0.0.jsonl"
if not train_file.exists():
    print(f"❌ ERREUR : Le fichier est introuvable ici : {train_file}")
else:
    print(f"✅ Fichier trouvé : {train_file.name}")

dataset_train = load_dataset("json", data_files=f"{DATASET_PATH}/sft_train_v1.0.0.jsonl", split="train")
dataset_val = load_dataset("json", data_files=f"{DATASET_PATH}/sft_val_v1.0.0.jsonl", split="train")
dataset_test = load_dataset("json", data_files=f"{DATASET_PATH}/sft_test_v1.0.0.jsonl", split="train")

if MODE_TEST:
    print("🧪 MODE TEST : Réduction des datasets...")
    dataset_train = dataset_train.select(range(50))
    dataset_val = dataset_val.select(range(10))
    dataset_test = dataset_test.select(range(10))

print("Chargement du tokenizer et des données...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# ==========================================
# 4. CHARGEMENT DU MODÈLE OPTIMISÉ POUR RTX 2060 (6GB VRAM)
# ==========================================
print("🧠 Chargement du modèle en 4-bit (BitsAndBytes)...")

# Configuration de compression extrême pour tenir sur 6Go de VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Préparation obligatoire quand on utilise la quantification 4-bit avec LoRA
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False # Requis pour le gradient checkpointing

# ==========================================
# 5. CONFIGURATION LORA
# ==========================================
print("🧠 CONFIGURATION LORA ...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] 
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==========================================
# 6. ARGUMENTS D'ENTRAÎNEMENT (CHECKPOINTS & MLFLOW)
# ==========================================
# Calcul dynamique des steps selon le mode
max_steps_run = 10 if MODE_TEST else 1000 
save_steps_run = 5 if MODE_TEST else 100
print("🧠  SFTConfig...")

# --- MAINTENANT (CORRIGÉ) ---
sft_config = SFTConfig(
    output_dir=output_dir,
    dataset_text_field="text",           # On le remet ICI (dans SFTConfig)
    max_length=768,                      # On teste 'max_length' au lieu de 'max_seq_length'
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    eval_strategy="steps",
    eval_steps=save_steps_run,
    save_strategy="steps",
    save_steps=save_steps_run,
    logging_steps=1,
    fp16=False,
    bf16=False,
    tf32=False,
    report_to="mlflow",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    remove_unused_columns=False,
)
# ==========================================
# 7. LANCEMENT DU SFT TRAINER
# ==========================================
print("🚀 Initialisation de l'entraînement 🚀")


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    processing_class=tokenizer,
    args=sft_config,                     # On passe l'objet sft_config
)
chemin_local = f"{output_dir}/modele_final_lora"
# Démarre l'entraînement au sein d'un "run" MLflow
with mlflow.start_run():
    trainer.train()
    trainer.model.save_pretrained(chemin_local)
    tokenizer.save_pretrained(chemin_local)
    print("🔥 Évaluation finale sur le dataset de TEST...")
    metrics = trainer.evaluate(eval_dataset=dataset_test, metric_key_prefix="test")
    mlflow.log_metrics(metrics)
    print(f"Métriques de test : {metrics}")
# ==========================================
# 8. SAUVEGARDE FINALE
# ==========================================
    print("Sauvegarde de l'adaptateur LoRA final...")

    chemin_local = f"{output_dir}/modele_final_lora"
    
    trainer.model.save_pretrained(chemin_local)
    tokenizer.save_pretrained(chemin_local)

    print("Enregistrement du modèle en tant qu'artefact dans MLflow...")

    mlflow.log_artifacts(chemin_local, artifact_path="modele_qwen_lora_final")

    print("Pipeline terminée ! Le modèle est sécurisé dans MLflow.")
    print("Pour voir les graphiques, tapez dans un autre terminal : mlflow ui")