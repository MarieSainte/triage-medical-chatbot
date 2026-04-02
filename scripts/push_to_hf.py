import os
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict

# 1. Charger les variables du fichier .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def push_to_hf(dataset_name, local_dir, repo_id):
    if not hf_token:
        print("❌ Erreur : HF_TOKEN non trouvé dans le fichier .env")
        return

    print(f"🚀 Préparation de l'upload pour {repo_id}...")
    
    # Chargement des fichiers locaux
    ds_dict = DatasetDict({
        "train": load_dataset("json", data_files=f"{local_dir}/{dataset_name}_train_v1.0.0.jsonl", split="train"),
        "validation": load_dataset("json", data_files=f"{local_dir}/{dataset_name}_val_v1.0.0.jsonl", split="train"),
        "test": load_dataset("json", data_files=f"{local_dir}/{dataset_name}_test_v1.0.0.jsonl", split="train")
    })
    
    # Envoi sur Hugging Face avec le token sécurisé
    ds_dict.push_to_hub(repo_id, token=hf_token)
    print(f"✅ {repo_id} est maintenant en ligne !")

def push_to_hf_metadata(dataset_name, local_dir, repo_id,file_with_metadata):
    if not hf_token:
        print("❌ Erreur : HF_TOKEN non trouvé dans le fichier .env")
        return

    print(f"🚀 Préparation de l'upload pour {repo_id}...")
    
    # Chargement des fichiers locaux
    ds_dict = DatasetDict({
        "train": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_train_v1.0.0.jsonl", split="train"),
        "validation": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_val_v1.0.0.jsonl", split="train"),
        "test": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_test_v1.0.0.jsonl", split="train"),
        "metadata": load_dataset("json", data_files=f"{local_dir}/{file_with_metadata}", split="train")
    })
    
    # Envoi sur Hugging Face avec le token sécurisé
    ds_dict.push_to_hub(repo_id, token=hf_token)
    print(f"✅ {repo_id} est maintenant en ligne !")

#push_to_hf("sft", "../data/data_versioned/sft", "huggingjojo/medical-bilingual-sft")