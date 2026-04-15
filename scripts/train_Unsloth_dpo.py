
import unsloth  
import os
import sys
import json
from pathlib import Path
import mlflow
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# 1. PARAMETRES ET MODE TEST
# ==========================================
MODE_TEST = False

BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = BASE_DIR / "data" / "data_versioned" / "dpo"
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

sft_adapter_path = str(BASE_DIR / "models" / "unsloth_sft_lora_2026-04-14_16-11")
output_dir = str(BASE_DIR / "models" / f"unsloth_dpo_lora_{date_str}")
base_model_id = "Qwen/Qwen3-1.7B-Base"

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\hf_cache"

print("[MLflow] Initialisation de MLflow...")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(f"Triage_DPO_POC_Unsloth_{date_str}")

print("[Dataset] Chargement du dataset DPO...")
train_dataset = load_dataset("json", data_files=str(data_dir / "dpo_train_v1.0.0.jsonl"), split="train")
val_dataset   = load_dataset("json", data_files=str(data_dir / "dpo_val_v1.0.0.jsonl"),   split="train")
test_dataset  = load_dataset("json", data_files=str(data_dir / "dpo_test_v1.0.0.jsonl"),  split="train")

if MODE_TEST:
    print("[TEST] MODE TEST : Reduction des datasets DPO...")
    train_dataset = train_dataset.select(range(20))
    val_dataset   = val_dataset.select(range(5))
    test_dataset  = test_dataset.select(range(5))

# ==========================================
# 2. CHARGEMENT DU MODELE (UNSLOTH)
# ==========================================
print("[Model] Chargement du modele SFT avec Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=sft_adapter_path,  
    max_seq_length=768,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# ==========================================
# 3. PATCH DPO + CONFIGURATION LORA
# ==========================================
print("[DPO] Patch DPO Unsloth...")
PatchDPOTrainer()

from peft import PeftModel
if isinstance(model, PeftModel) or hasattr(model, 'peft_config'):
    print("[LoRA] Adapters LoRA SFT detectes -> reutilisation directe pour DPO.")
else:
    print("[LoRA] Application des adapters LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

# ==========================================
# 3b. FORMATAGE DU DATASET (messages -> texte ChatML)
# ==========================================

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
    "{% endfor %}"
)
tokenizer.chat_template = CHATML_TEMPLATE


def serialize_messages(messages):
    """Serialise les contenus dict en JSON string pour apply_chat_template."""
    return [
        {
            "role": msg["role"],
            "content": json.dumps(msg["content"], ensure_ascii=False)
                       if isinstance(msg["content"], dict)
                       else msg["content"]
        }
        for msg in messages
    ]


def format_dpo_row(examples):
    """Convertit prompt/chosen/rejected (listes de messages) en texte ChatML.

    DPOTrainer attend prompt, chosen, rejected comme chaines de texte.
    - prompt  : system + user avec add_generation_prompt=True
    - chosen  : uniquement le tour assistant choisi (texte seul, sans prompt)
    - rejected: uniquement le tour assistant rejeté (texte seul, sans prompt)
    """
    batch_size = len(examples["prompt"])
    prompts, chosens, rejecteds = [], [], []

    for i in range(batch_size):
        prompt_text = tokenizer.apply_chat_template(
            serialize_messages(examples["prompt"][i]),
            tokenize=False,
            add_generation_prompt=True,
        )
        chosen_text = tokenizer.apply_chat_template(
            serialize_messages(examples["chosen"][i]),
            tokenize=False,
            add_generation_prompt=False,
        )
        rejected_text = tokenizer.apply_chat_template(
            serialize_messages(examples["rejected"][i]),
            tokenize=False,
            add_generation_prompt=False,
        )
        prompts.append(prompt_text)
        chosens.append(chosen_text)
        rejecteds.append(rejected_text)

    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}


print("[Dataset] Formatage ChatML des datasets...")
train_dataset = train_dataset.map(format_dpo_row, batched=True, remove_columns=train_dataset.column_names)
val_dataset   = val_dataset.map(format_dpo_row,   batched=True, remove_columns=val_dataset.column_names)
test_dataset  = test_dataset.map(format_dpo_row,  batched=True, remove_columns=test_dataset.column_names)

# ==========================================
# 4. CONFIGURATION DPO
# ==========================================
max_steps_run  = 10 if MODE_TEST else 60
save_steps_run = 5  if MODE_TEST else 20

print("[DPO] Configuration DPOConfig...")
dpo_config = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6, 
    eval_strategy="steps",
    eval_steps=save_steps_run,
    per_device_eval_batch_size=1,
    save_strategy="steps",
    save_steps=save_steps_run,
    save_total_limit=2,
    logging_steps=5,
    max_steps=max_steps_run,
    fp16=False,
    bf16=False,
    report_to="mlflow",
    remove_unused_columns=False,
    beta=0.1,
    max_length=768,
    push_to_hub=False,
)

print("[Train] Lancement de l'entrainement DPO...")
trainer = DPOTrainer(
    model=model,
    ref_model=None,  
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
)

with mlflow.start_run(run_name="Unsloth_DPO_Run"):
    trainer.train()

    print("[Eval] Evaluation finale sur le dataset de TEST...")
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test"
    )
    mlflow.log_metrics(test_metrics)
    print(f"Metriques de test : {test_metrics}")

    print("[Save] Sauvegarde finale...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    mlflow.log_artifacts(output_dir, artifact_path="modele_unsloth_dpo_final")

print("[Done] Entrainement DPO Unsloth termine.")
