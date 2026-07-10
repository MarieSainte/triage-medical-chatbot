
import unsloth  
import os
import sys
import json
from pathlib import Path
import mlflow
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, PatchDPOTrainer
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# 1. PARAMETRES ET MODE TEST
# ==========================================
MODE_TEST = False

# Aligne sur le SFT : 1280 couvre 97% du dataset (768 tronquait 30%).
MAX_SEQ_LENGTH = 1280

# racine du projet (le script est dans scripts/training/, deux niveaux sous la racine)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
data_dir = BASE_DIR / "data" / "data_versioned" / "dpo"
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

# Checkpoint SFT de depart. Surchargable via SFT_ADAPTER_PATH.
sft_adapter_path = os.getenv(
    "SFT_ADAPTER_PATH",
    str(BASE_DIR / "models" / "unsloth_sft_lora_2026-07-10_02-45"),
)
output_dir = str(BASE_DIR / "models" / f"unsloth_dpo_lora_{date_str}")
base_model_id = "Qwen/Qwen3-1.7B-Base"

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\hf_cache"

print("[MLflow] Initialisation de MLflow...")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(f"Triage_DPO_POC_Unsloth_{date_str}")

def load_dpo_jsonl(path: Path) -> Dataset:
    """Charge un JSONL DPO en serialisant les content dict en JSON (evite l'inference pyarrow)."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            for field in ("prompt", "chosen", "rejected"):
                for message in example[field]:
                    if isinstance(message["content"], dict):
                        message["content"] = json.dumps(message["content"], ensure_ascii=False)
            rows.append(
                {
                    "prompt": example["prompt"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"],
                }
            )
    return Dataset.from_list(rows)


print("[Dataset] Chargement du dataset DPO...")
train_dataset = load_dpo_jsonl(data_dir / "dpo_train_v2.0.0.jsonl")
val_dataset   = load_dpo_jsonl(data_dir / "dpo_val_v2.0.0.jsonl")
test_dataset  = load_dpo_jsonl(data_dir / "dpo_test_v2.0.0.jsonl")

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
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)

# EOS = <|im_end|> et pad != eos (memes correctifs que le SFT, forces ici par securite).
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "<|endoftext|>"
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# Les deux tokens de fin en stop (le fix embeddings les rend equivalents, greedy prend le plus petit id).
model.generation_config.eos_token_id = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|endoftext|>"),
]
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Meme fix embeddings que le SFT, pour partir des memes conditions. Jamais a l'inference/merge.
def fix_untrained_chatml_embeddings(model, tokenizer):
    import torch
    emb = model.get_input_embeddings().weight
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    with torch.no_grad():
        norms = emb.norm(dim=1).float()
        median = norms.median()
        trained_mask = norms > 0.5 * median
        mean_trained = emb[trained_mask].mean(dim=0)
        before = norms[im_end].item()
        emb[im_end] = emb[endoftext].clone()
        emb[im_start] = mean_trained.to(emb.dtype)
    print(f"[Fix] embeddings ChatML répares : "
          f"im_end {before:.3f} -> {emb[im_end].norm().item():.3f} (copie endoftext), "
          f"im_start -> moyenne des lignes entraînées")

fix_untrained_chatml_embeddings(model, tokenizer)

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
    """Convertit prompt/chosen/rejected en texte ChatML (pas d'add_generation_prompt : evite de dupliquer <|im_start|>assistant)."""
    batch_size = len(examples["prompt"])
    prompts, chosens, rejecteds = [], [], []

    for i in range(batch_size):
        prompt_text = tokenizer.apply_chat_template(
            serialize_messages(examples["prompt"][i]),
            tokenize=False,
            add_generation_prompt=False,
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


def prepare_dpo_eval_dataset(trainer, dataset, dataset_name):
    """Pre-tokenise un dataset DPO pour evaluate() (le collator attend prompt_ids/chosen_ids/rejected_ids)."""
    return trainer._prepare_dataset(
        dataset,
        trainer.processing_class,
        trainer.args,
        dataset_name,
    )


print("[Dataset] Formatage ChatML des datasets...")
train_dataset = train_dataset.map(format_dpo_row, batched=True, remove_columns=train_dataset.column_names)
val_dataset   = val_dataset.map(format_dpo_row,   batched=True, remove_columns=val_dataset.column_names)
test_dataset  = test_dataset.map(format_dpo_row,  batched=True, remove_columns=test_dataset.column_names)

# ==========================================
# 4. CONFIGURATION DPO
# ==========================================
max_steps_run  = 10 if MODE_TEST else 120
save_steps_run = 5  if MODE_TEST else 30

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
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    beta=0.1,
    # Pas de max_prompt_length dans cette version : on borne juste la sequence totale (prompts courts de toute facon).
    max_length=MAX_SEQ_LENGTH,
    max_seq_length=MAX_SEQ_LENGTH,
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
    test_eval_dataset = prepare_dpo_eval_dataset(trainer, test_dataset, "test")
    test_metrics = trainer.evaluate(
        eval_dataset=test_eval_dataset,
        metric_key_prefix="test"
    )
    mlflow.log_metrics(test_metrics)
    print(f"Metriques de test : {test_metrics}")

    print("[Save] Sauvegarde finale...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    mlflow.log_artifacts(output_dir, artifact_path="modele_unsloth_dpo_final")

print("[Done] Entrainement DPO Unsloth termine.")
