import torch
import mlflow
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer

# ==========================================
# 1. PARAMÈTRES ET MLFLOW
# ==========================================
model_id = "Qwen/Qwen3-1.7B-Base" # (Ou le chemin vers votre dossier local SFT si vous reprenez depuis vos poids SFT !)
output_dir = "../models/qwen-chsa-dpo-final"

print("📊 Initialisation de MLflow...")
mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("CHSA_Triage_DPO_POC")

# ==========================================
# 2. CHARGEMENT DONNÉES ET TOKENIZER
# ==========================================
print("📂 Chargement du dataset DPO formaté...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# On charge le fichier JSONL que vous avez créé à l'étape précédente
dataset = load_dataset("json", data_files="../data/data_versioned/dpo/dpo_train_v1.0.0.jsonl", split="train")

# ==========================================
# 3. CHARGEMENT DU MODÈLE (4-BIT)
# ==========================================
print("🧠 Chargement du modèle en 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# ==========================================
# 4. CONFIGURATION LORA
# ==========================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# On applique LoRA. Le DPOTrainer est assez intelligent pour savoir 
# qu'il doit désactiver LoRA pour calculer le score de référence !
model = get_peft_model(model, peft_config)

# ==========================================
# 5. PARAMÈTRES SPÉCIFIQUES DPO
# ==========================================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,      # Vital pour le DPO sur 6GB VRAM
    gradient_accumulation_steps=8,      
    gradient_checkpointing=True,        
    learning_rate=5e-5,                 # Le taux d'apprentissage doit être PLUS BAS qu'en SFT (sinon le modèle détruit sa grammaire)
    logging_steps=5,
    max_steps=500,                      
    fp16=True,
    report_to="mlflow",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False         # Obligatoire pour le DPOTrainer !
)

# ==========================================
# 6. LE DPO TRAINER
# ==========================================
print("🚀 Lancement de l'alignement DPO...")

trainer = DPOTrainer(
    model=model,
    ref_model=None,                     # On met None car PEFT gère le modèle de référence automatiquement
    args=training_args,
    beta=0.1,                           # La "température" du DPO (0.1 est le standard de l'industrie)
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_length=1024,                    # Longueur max totale (Prompt + Réponse)
    max_prompt_length=512               # Longueur max de la question seule
)

with mlflow.start_run(run_name="DPO_UltraMedical_Run1"):
    trainer.train()

# ==========================================
# 7. SAUVEGARDE ET REGISTRE MLFLOW
# ==========================================
    print("💾 Sauvegarde finale...")
    chemin_local = f"{output_dir}/modele_dpo_lora"
    trainer.model.save_pretrained(chemin_local)
    tokenizer.save_pretrained(chemin_local)

    print("📦 Enregistrement dans MLflow...")
    mlflow.log_artifacts(chemin_local, artifact_path="modele_medical_aligne")

    print("✅ Entraînement DPO terminé ! Vous avez un modèle aligné médicalement.")