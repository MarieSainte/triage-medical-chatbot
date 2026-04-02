from sklearn.model_selection import train_test_split
import os

VERSION = "v1.0.0"
SEED = 42

def create_triple_split(df, dataset_name):

    print("🏗️ Création des versions Train/Val/Test...")

    # 1. Séparer le Train (80%) du reste (20%)
    train_df, temp_df = train_test_split(
        df, test_size=0.20, random_state=SEED
    )
    
    # 2. Séparer le reste en Validation (50% de 20% = 10%) et Test (10%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=SEED
    )
    
    # Création du dossier de versioning
    base_path = f"../data/data_versioned/{dataset_name}"
    os.makedirs(base_path, exist_ok=True)
    
    # Sauvegardes
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
    
    for split_name, split_df in splits.items():
        file_path = f"{base_path}/{dataset_name}_{split_name}_{VERSION}.jsonl"
        split_df.to_json(file_path, orient='records', lines=True, force_ascii=False)
        print(f"💾 Sauvegardé : {file_path} ({len(split_df)} lignes)")

    print("\n✅ dataset anonymisé et versionné split : ok")

#create_triple_split(dataset_sft_qwen_final, "sft")