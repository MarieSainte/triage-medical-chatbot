
import unsloth  
import os
import sys
import json
from pathlib import Path
import mlflow
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from datetime import datetime
import torch
from unsloth.chat_templates import train_on_responses_only

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==========================================
# 1. PARAMETRES ET MODE TEST
# ==========================================
MODE_TEST = False

MAX_SEQ_LENGTH = 1280

# racine du projet (le script est dans scripts/training/, deux niveaux sous la racine)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned" / "sft"
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

output_dir = str(BASE_DIR / "models" / f"unsloth_sft_lora_{date_str}")
base_model_id = "Qwen/Qwen3-1.7B-Base"

os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_CACHE"] = r"D:\hf_cache"

print("[MLflow] Initialisation de MLflow...")
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(f"Triage_SFT_POC_Unsloth_{date_str}")

def load_sft_jsonl(path: Path) -> Dataset:
    """Charge un JSONL SFT en serialisant les content dict en JSON (evite l'inference pyarrow sur colonne mixte)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            for msg in ex["messages"]:
                if isinstance(msg["content"], dict):
                    msg["content"] = json.dumps(msg["content"], ensure_ascii=False)
            rows.append({"messages": ex["messages"]})
    return Dataset.from_list(rows)


print("[Dataset] Chargement du dataset SFT...")
dataset_train = load_sft_jsonl(DATASET_PATH / "sft_train_v2.0.0.jsonl")
dataset_val = load_sft_jsonl(DATASET_PATH / "sft_val_v2.0.0.jsonl")
dataset_test = load_sft_jsonl(DATASET_PATH / "sft_test_v2.0.0.jsonl")
print(f"[Dataset] train={len(dataset_train)} val={len(dataset_val)} test={len(dataset_test)}")

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
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    device_map="auto"
)

model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
model.generation_config.eos_token_id = [
    tokenizer.convert_tokens_to_ids("<|im_end|>"),
    tokenizer.convert_tokens_to_ids("<|endoftext|>"),
]
model.generation_config.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

def fix_untrained_chatml_embeddings(model, tokenizer):
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
    """Convertit les messages en texte ChatML (gere exemple unique ou batch, serialise le content assistant en JSON)."""
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


def prepare_responses_only_eval_dataset(trainer, dataset, dataset_name):
    """Rejoue le masquage responses_only manquant pour un dataset passe a evaluate()."""
    packing = trainer.args.packing if trainer.args.eval_packing is None else trainer.args.eval_packing
    prepared_dataset = trainer._prepare_dataset(
        dataset,
        trainer.processing_class,
        trainer.args,
        packing,
        formatting_func,
        dataset_name,
    )

    mask_responses_only = train_on_responses_only(
        trainer=None,
        tokenizer=trainer.processing_class,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
        return_function=True,
    )
    prepared_dataset = prepared_dataset.map(
        mask_responses_only,
        batched=True,
        desc=f"Masking assistant labels for {dataset_name} dataset",
    )
    prepared_dataset = prepared_dataset.filter(
        lambda example: any(label != -100 for label in example["labels"]),
        desc=f"Filtering fully masked rows from {dataset_name} dataset",
    )
    return prepared_dataset


if MODE_TEST:
    # Contrôle visuel : le texte doit contenir les 3 tours et finir par <|im_end|>
    print("[TEST] Exemple formaté (tronqué) :")
    print(formatting_func(dataset_train[0])[0][:500])
    print("   [...fin :]", repr(formatting_func(dataset_train[0])[0][-80:]))


# ==========================================
# 4. CONFIGURATION SFT
# ==========================================
max_steps_run = 10 if MODE_TEST else 1400
save_steps_run = 5 if MODE_TEST else 200

print("[SFT] Configuration SFTConfig...")
sft_config = SFTConfig(
    output_dir=output_dir,
    max_seq_length=MAX_SEQ_LENGTH,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    # batch 1 : l'eval convertit les logits en fp32 -> OOM sur 6 GiB avec le defaut TRL de 8.
    per_device_eval_batch_size=1,
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
    packing=False,
    padding_free=False,
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

# Loss sur les reponses assistant uniquement (system/user masques a -100).
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

with mlflow.start_run(run_name="Unsloth_SFT_Run"):
    trainer.train()

    print("[Eval] Evaluation finale sur le dataset de TEST...")
    eval_dataset = prepare_responses_only_eval_dataset(trainer, dataset_test, "test")

    test_results = trainer.evaluate(
        eval_dataset=eval_dataset,
        metric_key_prefix="test"
    )
    mlflow.log_metrics(test_results)
    print(f"Metriques de test : {test_results}")

    print("[Save] Sauvegarde finale...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    mlflow.log_artifacts(output_dir, artifact_path="modele_unsloth_sft_final")

print("[Done] Entrainement SFT Unsloth termine.")
