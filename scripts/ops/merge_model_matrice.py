import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen3-1.7B-Base"
HF_CACHE_DIR = "D:/hf_cache"
# Checkpoint LoRA a fusionner. Surchargable via LORA_PATH.
LORA_PATH = os.getenv("LORA_PATH", "./models/unsloth_dpo_lora_2026-04-24_21-44/checkpoint-60")
OUTPUT_DIR = os.getenv("MERGE_OUTPUT_DIR", "./models/qwen3-1.7b-dpo-merged")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Chargement du modèle de base en float32 (bf16/fp16 non supportes sur cette RTX) ---
print(f"[1/5] Chargement du modèle de base : {BASE_MODEL_ID} (float32)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",          # Fusion sur CPU pour éviter les problèmes VRAM/dtype
    cache_dir=HF_CACHE_DIR,
    trust_remote_code=True,
)

print(f"[2/5] Chargement du tokenizer depuis le checkpoint LoRA...")
tokenizer = AutoTokenizer.from_pretrained(
    LORA_PATH,
    trust_remote_code=True,
)

# Ne PAS reparer les embeddings ici : le modele fusionne s'arrete nativement. Modifier ici degraderait la lecture du prompt.

print(f"[3/5] Application du LoRA depuis : {LORA_PATH}...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

print(f"[4/5] Fusion des poids LoRA dans le modèle de base (merge_and_unload)...")
merged_model = model.merge_and_unload()

# --- Correctifs de configs avant sauvegarde ---
# 1. EOS = <|im_end|> : sans ce fix, vLLM/transformers ne stoppent jamais (Qwen3-Base a eos=<|endoftext|>).
print("[Fix] eos_token = <|im_end|>, pad = <|endoftext|>...")
tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "<|endoftext|>"
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
merged_model.config.eos_token_id = im_end_id
merged_model.config.pad_token_id = endoftext_id
# les deux en stop tokens : <|im_end|> + <|endoftext|>
merged_model.generation_config.eos_token_id = [im_end_id, endoftext_id]
merged_model.generation_config.pad_token_id = endoftext_id

# 3. Chat template ChatML COMPLET. Le template herite du checkpoint LoRA ne
# gere PAS add_generation_prompt (pas de prefixe assistant) : a l'inference, le
# modele "continue" le texte du user au lieu de repondre. On ecrit ici le
# template complet, identique au format d'entrainement, pour que le modele
# publie (HF -> vLLM -> gate CI) soit correct a la source, sans rustine runtime.
# NB : doit rester identique a la constante CHATML de test_CI/eval_model.py.
print("[Fix] chat_template ChatML complet (add_generation_prompt)...")
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)

print(f"[5/5] Sauvegarde du modèle fusionné dans : {OUTPUT_DIR}...")
merged_model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
tokenizer.save_pretrained(OUTPUT_DIR)

# 2. Compat transformers 4.x : duplique rope_theta et torch_dtype a la racine (un lecteur 4.x ne lit pas le format 5.x -> attention cassee).
config_path = os.path.join(OUTPUT_DIR, "config.json")
with open(config_path, encoding="utf-8") as f:
    cfg = json.load(f)
rope_params = cfg.get("rope_parameters") or {}
if "rope_theta" not in cfg and "rope_theta" in rope_params:
    cfg["rope_theta"] = rope_params["rope_theta"]
    print(f"[Fix] config.json : rope_theta={cfg['rope_theta']} dupliqué à la racine (compat 4.x)")
if "torch_dtype" not in cfg and "dtype" in cfg:
    cfg["torch_dtype"] = cfg["dtype"]
    print(f"[Fix] config.json : torch_dtype={cfg['torch_dtype']} dupliqué (compat 4.x)")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

print("[OK] Fusion terminee avec succes ! Le modele est pret.")
print(f"   -> Dossier de sortie : {os.path.abspath(OUTPUT_DIR)}")