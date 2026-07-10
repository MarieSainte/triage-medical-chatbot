import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

VERSION = "v1.0.0"
SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned"

INPUT_FILE = DATASET_PATH / "data_dpo_structured.jsonl"

out_dir = DATASET_PATH / "dpo"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Lecture : {INPUT_FILE}")
df = pd.read_json(INPUT_FILE, lines=True)
print(f"Total lignes : {len(df)}")

train_df, temp_df = train_test_split(df, test_size=0.20, random_state=SEED)
val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=SEED)

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out_path = out_dir / f"dpo_{split_name}_{VERSION}.jsonl"
    split_df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    print(f"Sauvegarde : {out_path} ({len(split_df)} lignes)")

print(f"\nSplit OK : train={len(train_df)} | val={len(val_df)} | test={len(test_df)}")
