"""Pousse un modèle local (dossier HF : config + safetensors + tokenizer) vers
un repo Hugging Face. Pendant du push_to_hf.py qui, lui, ne gère que les datasets.

Usage :
    python scripts/ops/push_model_to_hf.py                       # valeurs par défaut ci-dessous
    python scripts/ops/push_model_to_hf.py <dossier> <repo_id>   # override
"""
import os
import sys
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
hf_token = os.getenv("HF_TOKEN")


def push_model_to_hf(folder: str, repo_id: str, private: bool = True,
                     commit_message: str = "Upload model"):
    if not hf_token:
        print("Erreur : HF_TOKEN non trouvé dans le fichier .env")
        return
    if not os.path.isdir(folder):
        print(f"Erreur : dossier introuvable : {folder}")
        return

    api = HfApi(token=hf_token)
    print(f"Création/vérif du repo {repo_id} (private={private})...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"Upload de {folder} -> {repo_id} ...")
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"{repo_id} est maintenant en ligne sur Hugging Face !")


if __name__ == "__main__":
    folder  = sys.argv[1] if len(sys.argv) > 1 else "production_model"
    repo_id = sys.argv[2] if len(sys.argv) > 2 else "huggingjojo/medical-chatbot-model"
    push_model_to_hf(folder, repo_id)
