import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Configuration des chemins
base_model_id = "Qwen/Qwen3-1.7B-Base" # Ton modèle de base
lora_model_path = "./models/modele_final_lora" # <--- METS LE CHEMIN DE TON DOSSIER ICI

print(f"Loading model from {lora_model_path}...")

# 2. Chargement du Tokenizer
tokenizer = AutoTokenizer.from_pretrained(lora_model_path, trust_remote_code=True)

# 3. Chargement du modèle de base en 4-bit (pour ta 2060)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    trust_remote_code=True
)

# 4. Chargement des poids LoRA entraînés
model = PeftModel.from_pretrained(model, lora_model_path)
model.eval()

# 5. Fonction de test
def ask_medical_bot(prompt):
    # On utilise le format de chat de Qwen si tu as entraîné en mode chat
    messages = [
        {"role": "system", "content": "Tu es un assistant médical d'accueil et de triage strict. Ton rôle est d'évaluer l'urgence et d'orienter le patient. RÈGLE ABSOLUE : Tu ne dois JAMAIS prescrire ou suggérer de médicaments ou de traitements."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\n--- 🔍 FORMAT ENVOYÉ AU MODÈLE ---")
    print(text)
    print("----------------------------------\n")
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>")
    ]
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.1, 
            top_p=0.85,
            do_sample=True,
            eos_token_id=terminators, 
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response.strip()

# --- TEST ---
print("\n--- TEST DU MODÈLE ---")
#test_prompt = "J'ai hyper mal à la poitrine depuis une demi-heure, ça me serre super fort et la douleur descend dans mon bras gauche. J'ai aussi un peu de mal à respirer et je transpire beaucoup alors que je n'ai rien fait. J'ai très peur."
#test_prompt = "Bonjour, j'ai une douleur très très forte en bas à droite de mon ventre qui a commencé hier soir. J'ai vomi deux fois ce matin et je me sens hyper faible. J'ai pris ma température et j'ai 38.5. Qu'est-ce que je dois faire ?"
#test_prompt = "Coucou, j'ai le nez qui coule comme une fontaine depuis 3 jours et je tousse un peu, surtout quand je me couche. Je n'ai pas de fièvre mais j'ai la gorge qui gratte beaucoup. Est-ce que je dois aller voir un médecin ?"
test_prompt = "Je me sens vraiment pas bien du tout aujourd'hui, j'ai la tête qui tourne et j'ai mal partout. J'ai pris un doliprane mais ça passe pas."
print(f"Question: {test_prompt}")
print(f"Réponse du Bot:\n{ask_medical_bot(test_prompt)}")