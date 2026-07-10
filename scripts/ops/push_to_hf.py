import os
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def push_to_hf(dataset_name, local_dir, repo_id, version):
    if not hf_token:
        print("Erreur : HF_TOKEN non trouvé dans le fichier .env")
        return

    print(f"Préparation de l'upload pour {repo_id} (version {version})...")
    
    ds_dict = DatasetDict({
        "train": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_train_{version}.jsonl", split="train"),
        "validation": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_val_{version}.jsonl", split="train"),
        "test": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_test_{version}.jsonl", split="train")
    })
    
    ds_dict.push_to_hub(repo_id, token=hf_token)
    print(f"{repo_id} est maintenant en ligne sur Hugging Face !")

if __name__ == "__main__":
    push_to_hf("sft", "data/data_versioned", "huggingjojo/medical-bilingual-sft", "v2.0.0")