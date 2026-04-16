import os
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

def push_to_hf(dataset_name, local_dir, repo_id):
    if not hf_token:
        print("Erreur : HF_TOKEN non trouvé dans le fichier .env")
        return

    print(f"Préparation de l'upload pour {repo_id}...")
    
    ds_dict = DatasetDict({
        "train": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_train_v1.0.0.jsonl", split="train"),
        "validation": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_val_v1.0.0.jsonl", split="train"),
        "test": load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_test_v1.0.0.jsonl", split="train")
    })
    

    ds_dict.push_to_hub(repo_id, token=hf_token)
    print(f"{repo_id} est maintenant en ligne !")

def push_to_hf_metadata(dataset_name, local_dir, repo_id,file_with_metadata):
    if not hf_token:
        print("Erreur : HF_TOKEN non trouvé dans le fichier .env")
        return

    print(f"Préparation de l'upload pour {repo_id}...")
    
    train_ds = load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_train_v1.0.0.jsonl", split="train")
    val_ds = load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_val_v1.0.0.jsonl", split="train")
    test_ds = load_dataset("json", data_files=f"{local_dir}/{dataset_name}/{dataset_name}_test_v1.0.0.jsonl", split="train")
    metadata_ds = load_dataset("json", data_files=f"{local_dir}/{file_with_metadata}", split="train")
    print("train features:", train_ds.features)
    print("validation features:", val_ds.features)
    print("test features:", test_ds.features)
    print("metadata features:", metadata_ds.features)
    ds_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
        #"metadata": metadata_ds
    })

    
    ds_dict.push_to_hub(repo_id, token=hf_token)
    print(f"{repo_id} est maintenant en ligne !")

push_to_hf_metadata("sft", "data/data_versioned/", "huggingjojo/medical-bilingual-sft", "chsa_sft_bilingual_anonymized_v1.0.0_20260401.jsonl")