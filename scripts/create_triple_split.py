import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

VERSION = "v2.0.0"
SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned"


def create_triple_split(input_filename, dataset_name):

    input_path = DATASET_PATH / f"{input_filename}.jsonl"
    print(f"Lecture : {input_path}")
    df = pd.read_json(input_path, lines=True)
    print(f"Total lignes : {len(df)}")

    print("Creation des versions Train/Val/Test...")

    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED)

    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)

    out_dir = DATASET_PATH / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, split_df in splits.items():
        file_path = out_dir / f"{dataset_name}_{split_name}_{VERSION}.jsonl"
        split_df.to_json(file_path, orient="records", lines=True, force_ascii=False)
        print(f"Sauvegarde : {file_path} ({len(split_df)} lignes)")

    print(f"\nDataset split OK : train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")


create_triple_split("data_sft_v1.0.0_reviewed", "sft")
