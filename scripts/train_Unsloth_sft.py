
import unsloth  
import os
import sys
import json
from pathlib import Path
import mlflow
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# 1. PARAMETRES ET MODE TEST
# ==========================================
MODE_TEST = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned" / "sft"
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

output_dir = str(BASE_DIR / "models" / f"unsloth_sft_lora_{date_str}")
base_model_id = "Qwen/Qwen3-1.7B-Base"

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\hf_cache"

print("[MLflow] Initialisation de MLflow...")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(f"Triage_SFT_POC_Unsloth_{date_str}")

print("[Dataset] Chargement du dataset SFT...")
dataset_train = load_dataset("json", data_files=str(DATASET_PATH / "sft_train_v2.0.0.jsonl"), split="train")
dataset_val = load_dataset("json", data_files=str(DATASET_PATH / "sft_val_v2.0.0.jsonl"), split="train")
dataset_test = load_dataset("json", data_files=str(DATASET_PATH / "sft_test_v2.0.0.jsonl"), split="train")

if MODE_TEST:
    print("[TEST] MODE TEST : Reduction des datasets SFT...")
    dataset_train = dataset_train.select(range(50))
    dataset_val = dataset_val.select(range(10))
    dataset_test = dataset_test.select(range(10))

# ==========================================
# 2. CHARGEMENT DU MODELE (UNSLOTH)
# ==========================================
print("[Model] Chargement du modele avec Unsloth...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=768,
    dtype=None,
    load_in_4bit=True,
    device_map="auto"
)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

print("[LoRA] Configuration LoRA...")
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
# 3. FORMATAGE DU DATASET (format messages -> texte)
# ==========================================
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}"
    "{% endfor %}"
)

tokenizer.chat_template = CHATML_TEMPLATE


def formatting_func(examples):
    """Convertit les conversations messages en texte ChatML pour l'entrainement.

    Unsloth appelle cette fonction de deux facons :
    - Avec un exemple unique : examples["messages"] = [{role, content}, ...]
    - Avec un batch : examples["messages"] = [[{role, content}, ...], ...]

    Le contenu assistant est un dict structure {type, urgence, analyse, question}.
    On le serialise en JSON string avant d'appliquer le template ChatML.
    """
    messages_field = examples["messages"]
    if isinstance(messages_field, list) and len(messages_field) > 0 and isinstance(messages_field[0], dict):
        conversations = [messages_field]
    else:
        conversations = messages_field

    texts = []
    for messages in conversations:
        processed = [
            {
                "role": msg["role"],
                "content": json.dumps(msg["content"], ensure_ascii=False)
                           if isinstance(msg["content"], dict)
                           else msg["content"]
            }
            for msg in messages
        ]
        text = tokenizer.apply_chat_template(
            processed,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return texts


# ==========================================
# 4. CONFIGURATION SFT
# ==========================================
max_steps_run = 10 if MODE_TEST else 1400
save_steps_run = 5 if MODE_TEST else 100

print("[SFT] Configuration SFTConfig...")
sft_config = SFTConfig(
    output_dir=output_dir,
    max_seq_length=768,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_steps=30,
    eval_strategy="steps",
    eval_steps=save_steps_run,
    save_strategy="steps",
    save_steps=save_steps_run,
    logging_steps=5,
    max_steps=max_steps_run,
    fp16=False,
    bf16=False,
    report_to="mlflow",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    remove_unused_columns=False,
    packing=True,
    push_to_hub=False,
    hub_model_id=None,
    hub_token=None,
)

print("[Train] Lancement de l'entrainement SFT...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    processing_class=tokenizer,
    args=sft_config,
    formatting_func=formatting_func,
)

with mlflow.start_run(run_name="Unsloth_SFT_Run"):
    trainer.train()

    print("[Eval] Evaluation finale sur le dataset de TEST...")
    test_results = trainer.evaluate(
        eval_dataset=trainer._prepare_dataset(
            dataset_test,
            trainer.processing_class,
            trainer.args,
            trainer.args.packing,
            formatting_func, 
            "test"
        ),
        metric_key_prefix="test"
    )
    mlflow.log_metrics(test_results)
    print(f"Metriques de test : {test_results}")

    print("[Save] Sauvegarde finale...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    mlflow.log_artifacts(output_dir, artifact_path="modele_unsloth_sft_final")

print("[Done] Entrainement SFT Unsloth termine.")
