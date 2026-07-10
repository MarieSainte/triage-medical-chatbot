import sys
import json
import random
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "data_versioned"
INPUT_FILE = DATASET_PATH / "data_sft_v1.0.0_reviewed.jsonl"

def inspect_dataset(n_samples=10):
    print("Process ...")
    lines = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                lines.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️ Ligne invalide : {e}")

    print(f"\n📦 Total lignes : {len(lines)}\n")
    print("=" * 60)

    # Échantillon aléatoire
    samples = random.sample(lines, min(n_samples, len(lines)))

    for i, data in enumerate(samples, 1):
        print(f"\n🔹 Exemple {i}")
        messages = data.get("messages", [])

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                print(f"  [system] {content[:80]}...")

            elif role == "user":
                print(f"  [user]   {content[:100]}")

            elif role == "assistant":
                if isinstance(content, dict):
                    print(f"  [assistant]")
                    print(f"    type    : {content.get('type')}")
                    print(f"    urgence : {content.get('urgence')}")
                    print(f"    question: {content.get('question')}")
                    analyse = content.get('analyse') or ""
                    print(f"    analyse : {analyse[:100]}")
                else:
                    print(f"  [assistant] ⚠️ contenu non structuré : {str(content)[:100]}")

        print("-" * 60)

    # Stats globales
    print("\n📊 Stats")
    total_final = 0
    total_question = 0
    urgence_counter = {"Haute": 0, "Moyenne": 0, "Faible": 0, "None": 0}
    non_structured = 0

    for data in lines:
        for msg in data.get("messages", []):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if not isinstance(content, dict):
                    non_structured += 1
                    continue
                t = content.get("type")
                if t == "final":
                    total_final += 1
                    u = str(content.get("urgence"))
                    urgence_counter[u if u in urgence_counter else "None"] += 1
                elif t == "question":
                    total_question += 1

    print(f"  final    : {total_final}")
    print(f"  question : {total_question}")
    print(f"  urgences : {urgence_counter}")
    if non_structured:
        print(f"assistant non structurés : {non_structured}")

if __name__ == "__main__":
    print("Starting ...")
    inspect_dataset(n_samples=10)
    print("Ending ! ")