import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
output_dir = str(BASE_DIR / "models")

base_model_id = "Qwen/Qwen3-1.7B-Base"
adapter_path = str(Path(output_dir) / "modele_final_lora")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

def chat_loop(model=model, tokenizer=tokenizer, max_new_tokens=256, max_length=1024):
    print("-- Chat médical\n")

    system_prompt = f"""Tu es un médecin urgentiste et régulateur expert. 
        Ton rôle est d'analyser rapidement des situations cliniques pour effectuer un triage précis 
        (déterminer le niveau d'urgence et extraire les symptômes clés),
        Ton raisonnement et ton verdict médical final doivent être structurés au format JSON sous la balise ### ANALYSE.
        le JSON doit obligatoirement avoir 3 champs : urgence (haute, moyen, faible), analyse (justification du niveau d'urgence et recommendation courte, 
        Attention ne pose pas de question dans l'analyse), symptomes (la liste des symptomes identifié)
        Si tu dois poser une question pour établir un diagnostic utilise la balise ### ASSISTANT sans format JSON mais privilégie l'analyse. 
        ATTENTION tu dois soit analyser avec ### ANALYSE soit poser une question avec ### ASSISTANT pas les deux.
        si le cas de l'utilisateur te permet d'établir un niveau d'urgence alors choisi l'analyse sinon pose  une question mais jamais les deux.  
        tu ne dois jamais inventer les réponses des user
        
        voila des cas exemples :
        """

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        user_input = input("-- User : ")

        if user_input.lower() in ["exit", "quit"]:
            print("-- Fin du chat")
            break

        messages.append({
            "role": "user",
            "content": f"### USER\n{user_input}"
        })

        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += msg["content"] + "\n"
            elif msg["role"] == "user":
                prompt += msg["content"] + "\n"
            elif msg["role"] == "assistant":
                prompt += msg["content"] + "\n"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"-- Bot : {response}\n")

        messages.append({
            "role": "assistant",
            "content": response
        })

chat_loop()