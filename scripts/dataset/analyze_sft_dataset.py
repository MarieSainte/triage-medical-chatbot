"""Statistiques descriptives du dataset SFT (pour la présentation).

Analyse les 3 splits (train/val/test) :
- nombre d'exemples et de tours par split
- répartition des réponses par type (question / final) et par urgence
- répartition par metadata : langue de la source (fr/en), corpus d'origine
  (type_document), tag_origine, niveau de confiance
- longueurs des contents (caractères et mots) : symptômes user et réponses assistant
- longueur des séquences ChatML en tokens (si le tokenizer local est disponible)
  et % d'exemples qui tiennent dans max_seq_length=768

Usage :
    python scripts/dataset/analyze_sft_dataset.py            # rapport console
    python scripts/dataset/analyze_sft_dataset.py --md       # + docs/dataset_sft_stats.md
"""

import json
import re
import sys
import statistics
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SFT_DIR = BASE_DIR / "data" / "data_versioned" / "sft"
VERSION = "v2.0.0"
SPLITS = ["train", "val", "test"]
MAX_SEQ_LENGTH = 768  # celui de train_Unsloth_sft.py

# Tokenizer local (optionnel) : celui du modèle de prod, format ChatML identique au train
TOKENIZER_PATH = BASE_DIR / "production_model" / "tokenizer.json"


def load_tokenizer():
    try:
        from tokenizers import Tokenizer
        if TOKENIZER_PATH.exists():
            return Tokenizer.from_file(str(TOKENIZER_PATH))
    except ImportError:
        pass
    return None


def to_chatml(messages: list) -> str:
    """Reproduit le CHATML_TEMPLATE de train_Unsloth_sft.py."""
    parts = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        parts.append(f"<|im_start|>{msg['role']}\n{content}<|im_end|>\n")
    return "".join(parts)


def parse_metadata(raw) -> dict:
    """Extrait les champs utiles de la metadata.

    La metadata est stockée en repr Python (quotes simples, array(...) numpy),
    pas en JSON — on extrait par regex les champs scalaires.
    """
    if not isinstance(raw, str):
        raw = str(raw)
    fields = {}
    for key in ("tag_origine", "type_document"):
        m = re.search(rf"'{key}':\s*'([^']*)'", raw)
        fields[key] = m.group(1).replace("\\/", "/") if m else "<absent>"
    m = re.search(r"'niveau_confiance':\s*(\d+)", raw)
    fields["niveau_confiance"] = m.group(1) if m else "<absent>"
    fields["langue"] = derive_langue(fields["tag_origine"], fields["type_document"])
    return fields


def derive_langue(tag: str, doc: str) -> str:
    """Langue de la SOURCE (les exemples finaux sont tous en français) :
    - tags Mistral : 'francaise'/'française' vs 'anglaise'
    - corpus : MediQAl (ANR-MALADES) = français, MedQuad = anglais
    """
    tag_low = tag.lower()
    if "francaise" in tag_low or "française" in tag_low:
        return "fr"
    if "anglaise" in tag_low:
        return "en"
    if "MediQAl" in doc:
        return "fr"
    if "MedQuad" in doc:
        return "en"
    return "inconnu"


def describe(values: list) -> dict:
    if not values:
        return {"min": 0, "moy": 0, "med": 0, "max": 0}
    return {
        "min": min(values),
        "moy": round(statistics.mean(values), 1),
        "med": statistics.median(values),
        "max": max(values),
    }


def analyze_split(path: Path, tokenizer) -> dict:
    stats = {
        "exemples": 0,
        "tours_assistant": 0,
        "multi_tours": 0,
        "types": Counter(),
        "urgences": Counter(),
        "langues": Counter(),
        "docs": Counter(),
        "tags": Counter(),
        "confiances": Counter(),
        "len_user_chars": [],
        "len_user_mots": [],
        "len_assistant_chars": [],
        "len_question_chars": [],
        "len_analyse_chars": [],
        "len_seq_tokens": [],
    }

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            messages = ex["messages"]
            stats["exemples"] += 1

            meta = parse_metadata(ex.get("metadata", ""))
            stats["langues"][meta["langue"]] += 1
            stats["docs"][meta["type_document"]] += 1
            stats["tags"][meta["tag_origine"]] += 1
            stats["confiances"][meta["niveau_confiance"]] += 1

            n_assistant = sum(1 for m in messages if m["role"] == "assistant")
            stats["tours_assistant"] += n_assistant
            if n_assistant > 1:
                stats["multi_tours"] += 1

            for msg in messages:
                content = msg["content"]
                if msg["role"] == "user":
                    stats["len_user_chars"].append(len(content))
                    stats["len_user_mots"].append(len(content.split()))
                elif msg["role"] == "assistant":
                    data = content if isinstance(content, dict) else None
                    if data is None:
                        try:
                            data = json.loads(content)
                        except (json.JSONDecodeError, TypeError):
                            data = {}
                    raw = json.dumps(data, ensure_ascii=False) if data else str(content)
                    stats["len_assistant_chars"].append(len(raw))

                    t = data.get("type", "inconnu") if data else "inconnu"
                    stats["types"][t] += 1
                    if t == "final":
                        stats["urgences"][data.get("urgence") or "null"] += 1
                        if data.get("analyse"):
                            stats["len_analyse_chars"].append(len(data["analyse"]))
                    elif t == "question" and data.get("question"):
                        stats["len_question_chars"].append(len(data["question"]))

            if tokenizer is not None:
                stats["len_seq_tokens"].append(
                    len(tokenizer.encode(to_chatml(messages)).ids)
                )

    return stats


def fmt_dist(counter: Counter, total: int) -> str:
    return " / ".join(
        f"{k}: {v} ({100 * v / total:.1f}%)" for k, v in counter.most_common()
    )


def main():
    tokenizer = load_tokenizer()
    if tokenizer is None:
        print("[Info] Tokenizer local indisponible (pip install tokenizers) : stats tokens ignorées.")

    lines = []
    out = lines.append
    out(f"# Statistiques dataset SFT {VERSION}")
    out("")

    totals = {
        "exemples": 0,
        "types": Counter(),
        "urgences": Counter(),
        "langues": Counter(),
        "docs": Counter(),
        "tags": Counter(),
        "confiances": Counter(),
    }
    all_tokens = []

    for split in SPLITS:
        path = SFT_DIR / f"sft_{split}_{VERSION}.jsonl"
        if not path.exists():
            print(f"[Erreur] Fichier introuvable : {path}")
            sys.exit(1)
        s = analyze_split(path, tokenizer)
        totals["exemples"] += s["exemples"]
        for key in ("types", "urgences", "langues", "docs", "tags", "confiances"):
            totals[key] += s[key]
        all_tokens.extend(s["len_seq_tokens"])

        out(f"## Split {split} — {s['exemples']} exemples")
        out("")
        out(f"- Tours assistant : {s['tours_assistant']}"
            f" (conversations multi-tours : {s['multi_tours']})")
        out(f"- Types de réponse : {fmt_dist(s['types'], s['tours_assistant'])}")
        n_final = sum(s["urgences"].values())
        if n_final:
            out(f"- Urgences (type=final) : {fmt_dist(s['urgences'], n_final)}")
        out(f"- Langue de la source : {fmt_dist(s['langues'], s['exemples'])}")
        out(f"- Corpus d'origine : {fmt_dist(s['docs'], s['exemples'])}")
        out(f"- Tags d'origine : {fmt_dist(s['tags'], s['exemples'])}")
        out(f"- Niveau de confiance : {fmt_dist(s['confiances'], s['exemples'])}")
        out("")
        out("| Longueur (caractères) | min | moyenne | médiane | max |")
        out("|---|---|---|---|---|")
        for label, key in [
            ("Symptômes (user)", "len_user_chars"),
            ("Réponse assistant (JSON)", "len_assistant_chars"),
            ("Question posée", "len_question_chars"),
            ("Analyse (verdict final)", "len_analyse_chars"),
        ]:
            d = describe(s[key])
            out(f"| {label} | {d['min']} | {d['moy']} | {d['med']} | {d['max']} |")
        d = describe(s["len_user_mots"])
        out(f"| Symptômes (user), en mots | {d['min']} | {d['moy']} | {d['med']} | {d['max']} |")

        if s["len_seq_tokens"]:
            d = describe(s["len_seq_tokens"])
            sous_max = sum(1 for t in s["len_seq_tokens"] if t <= MAX_SEQ_LENGTH)
            out(f"| Séquence ChatML complète, en tokens | {d['min']} | {d['moy']} | {d['med']} | {d['max']} |")
            out("")
            out(f"- Exemples ≤ {MAX_SEQ_LENGTH} tokens (max_seq_length du train) : "
                f"{sous_max}/{s['exemples']} ({100 * sous_max / s['exemples']:.1f}%)")
        out("")

    out(f"## Global — {totals['exemples']} exemples")
    out("")
    n_tours = sum(totals["types"].values())
    out(f"- Types de réponse : {fmt_dist(totals['types'], n_tours)}")
    n_final = sum(totals["urgences"].values())
    if n_final:
        out(f"- Urgences (type=final) : {fmt_dist(totals['urgences'], n_final)}")
    out(f"- Langue de la source : {fmt_dist(totals['langues'], totals['exemples'])}")
    out(f"- Corpus d'origine : {fmt_dist(totals['docs'], totals['exemples'])}")
    out(f"- Tags d'origine : {fmt_dist(totals['tags'], totals['exemples'])}")
    out(f"- Niveau de confiance : {fmt_dist(totals['confiances'], totals['exemples'])}")
    if all_tokens:
        d = describe(all_tokens)
        sous_max = sum(1 for t in all_tokens if t <= MAX_SEQ_LENGTH)
        out(f"- Tokens par séquence : moy {d['moy']}, méd {d['med']}, max {d['max']} — "
            f"≤ {MAX_SEQ_LENGTH} tokens : {100 * sous_max / len(all_tokens):.1f}%")

    report = "\n".join(lines)
    print(report)

    if "--md" in sys.argv:
        md_path = BASE_DIR / "docs" / "dataset_sft_stats.md"
        md_path.write_text(report + "\n", encoding="utf-8")
        print(f"\n[Save] Rapport écrit dans {md_path}")


if __name__ == "__main__":
    main()
