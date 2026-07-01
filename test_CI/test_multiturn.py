import os
import sys

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# S'assurer qu'on utilise le modèle de production
MODEL_ID = "production_model"

print("Chargement du modele et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.eval()

# Import du prompt optimisé DSPy
try:
    sys.path.append(str(os.path.dirname(os.path.dirname(__file__))))
    from api.dspy.signatures import OPTIMIZED_SYSTEM_PROMPT
    SYSTEM_PROMPT = OPTIMIZED_SYSTEM_PROMPT
except Exception:
    SYSTEM_PROMPT = (
        "Tu es un medecin urgentiste charge de trier des situations cliniques.\n"
        "Reponds UNIQUEMENT en JSON strict, sans texte avant ni apres :\n"
        "{\"type\":\"final\",\"question\":null,\"urgence\":\"Haute|Moyenne|Faible\",\"analyse\":\"...\"}\n"
        "ou\n"
        "{\"type\":\"question\",\"question\":\"...\",\"urgence\":null,\"analyse\":null}"
    )

def extract_json(raw: str):
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

def generate_response(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop_ids = [tokenizer.eos_token_id, im_end_id]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )
    
    input_len = inputs["input_ids"].shape[1]
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response, extract_json(response)

# --- Test Multi-Turn ---
print("\n--- DEBUT DU TEST MULTI-TURN ---")

# Tour 1
user_input_1 = "Mon bébé de 3 mois pleure sans arrêt depuis tout à l'heure, je suis inquiète."
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": user_input_1}
]

print(f"\n[Patient] : {user_input_1}")
print("-> Inférence Tour 1 en cours...")
raw_1, data_1 = generate_response(messages)
print(f"[Modèle (brut)] : {raw_1}")

if data_1 and data_1.get("type") == "question":
    question = data_1.get("question")
    print(f"\n[Modèle JSON] -> Pose une question : {question}")
    
    # Ajout à l'historique
    messages.append({"role": "assistant", "content": raw_1})
    
    # Tour 2 : On répond à la question
    user_input_2 = "Il a 40°C de fièvre et il est tout mou, il ne réagit presque plus quand je lui parle."
    messages.append({"role": "user", "content": user_input_2})
    
    print(f"\n[Patient] : {user_input_2}")
    print("-> Inférence Tour 2 en cours...")
    raw_2, data_2 = generate_response(messages)
    print(f"[Modèle (brut)] : {raw_2}")
    
    if data_2 and data_2.get("type") == "final":
        urgence = data_2.get("urgence")
        analyse = data_2.get("analyse")
        print(f"\n[Modèle JSON] -> Décision Finale : URGENCE {urgence}")
        print(f"Analyse : {analyse}")
    else:
        print("\n[ECHEC] Le modèle n'a pas rendu de décision finale.")
else:
    print("\n[Modèle JSON] -> A rendu une décision directe (pas de question).")
