import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_dataset import DATASET
print("Dataset charge.")

# ----------------------------------- config
LIMIT      = int(os.getenv("EVAL_SAMPLE_LIMIT", "5"))   # 5 en CI, 24 en local
MODEL_ID   = os.getenv("MODEL_ID", "production_model")
CASES      = DATASET[:LIMIT]
LABELS     = ["Haute", "Moyenne", "Faible", "question"]
URGENCES   = LABELS[:3]
THRESHOLDS = {"recall_haute": 0.90, "accuracy": 0.70}   # asymetrique : securite d'abord

CHATML = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)

# Prompt DSPy compile : meme JSON que celui charge par l'API -> parite prod
_json_path = (Path(__file__).resolve().parent.parent / "api" / "dspy"
              / f"dspy_optimized_triage_{os.getenv('DSPY_PROFILE', 'sft').lower()}.json")
try:
    _cfg = json.loads(_json_path.read_text(encoding="utf-8-sig"))
    SYSTEM_PROMPT = _cfg["system_prompt"]
    DEMOS = _cfg.get("demos", [])
except Exception as e:
    sys.exit(f"ERREUR : prompt DSPy de prod illisible ({_json_path}) — {e}")
print("Prompt optimise charge (JSON DSPy).")
if "Exemple" in SYSTEM_PROMPT:
    DEMOS = [] 


# --------------------------------- modele
class Model:
    """Modele fusionne, charge sur CPU."""

    def __init__(self, model_id: str = MODEL_ID):
        self._fix_bom(model_id)
        print(f"Chargement du modele {model_id} sur CPU...")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tpl = getattr(self.tok, "chat_template", None) or ""
        if "add_generation_prompt" not in tpl or "<|im_start|>assistant" not in tpl:
            self.tok.chat_template = CHATML
            print("Chat template ChatML normalise pour l'inference.")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cpu",
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        self.model.eval()
        self.stop_ids = list({
            self.tok.eos_token_id,
            self.tok.convert_tokens_to_ids("<|im_end|>"),
            self.tok.convert_tokens_to_ids("<|endoftext|>"),
        })

    @staticmethod
    def _fix_bom(model_id: str):
        """Retire un eventuel BOM UTF-8 des JSON d'un modele local."""
        path = Path(model_id)
        if not path.is_dir():
            return
        for p in path.glob("*.json"):
            raw = p.read_bytes()
            if raw.startswith(b"\xef\xbb\xbf"):
                p.write_text(raw.decode("utf-8-sig"), encoding="utf-8")
                print(f"BOM UTF-8 retire de : {p.name}")

    def _messages(self, symptomes: str) -> List[Dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for d in DEMOS:
            if d.get("symptomes") and d.get("reponse"):
                msgs += [{"role": "user", "content": d["symptomes"]},
                         {"role": "assistant", "content": d["reponse"]}]
        return msgs + [{"role": "user", "content": symptomes}]

    def _generate(self, messages: List[Dict], max_new: int = 256) -> Dict:
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False, 
                pad_token_id=self.tok.eos_token_id, eos_token_id=self.stop_ids,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        raw = self.tok.decode(gen, skip_special_tokens=True).strip()

        clean = len(gen) < max_new and gen[-1].item() in self.stop_ids
        data = extract_json(raw)
        if not data:
            print(f"  Pas de JSON valide : {raw[:100]}...")
            return {"type": "error", "message": "No JSON found", "_raw": raw, "_clean_stop": clean}
        if data.get("type") == "final":
            u = str(data.get("urgence", "")).capitalize()
            data["urgence"] = u if u in URGENCES else "Inconnue"
        data.update(_raw=raw, _clean_stop=clean)
        return data

    def predict(self, symptomes: str) -> Dict:
        return self._generate(self._messages(symptomes))

    def predict_turn2(self, symptomes: str, turn1: Dict, followup: str) -> Dict:
        """Tour 2 : reponse BRUTE du tour 1 + relance patient dans le contexte."""
        raw1 = turn1.get("_raw") or json.dumps(
            {k: v for k, v in turn1.items() if k != "_raw"}, ensure_ascii=False)
        return self._generate(self._messages(symptomes) + [
            {"role": "assistant", "content": raw1},
            {"role": "user", "content": followup},
        ])


# ----------------------------------------------------------------- helpers
def extract_json(raw: str) -> Optional[Dict]:
    """Extrait le premier objet JSON, tolerant au texte parasite apres."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()).strip()
    start = raw.find("{")
    if start == -1:
        return None
    for end in range(len(raw), start, -1):
        try:
            return json.loads(raw[start:end])
        except Exception:
            continue
    return None


def label_of(pred: Dict) -> str:
    if pred.get("type") == "question":
        return "question"
    u = pred.get("urgence", "Inconnue")
    return u if u in URGENCES else "Autre"


# --------------------------------------- evaluation
def evaluate():
    model = Model()
    conf = {a: {p: 0 for p in LABELS} for a in LABELS}
    n = len(CASES)
    correct = missing = multiturn = gen_total = gen_stopped = 0
    haute_total = haute_ok = flow_total = flow_q1 = flow_ok = 0

    print(f"\nEvaluation multi-tour sur {n} exemples (limite={LIMIT})")
    print("=" * 60)

    for i, case in enumerate(CASES, 1):
        print(f"[{i}/{n}] {case['input'][:60]}...")
        pred = model.predict(case["input"])
        turns = 1
        gen_total += 1
        gen_stopped += bool(pred.get("_clean_stop"))

        expects_q1 = case.get("expected_turn1_type") == "question"
        if expects_q1:
            flow_total += 1
            if pred.get("type") == "question":
                flow_q1 += 1
            else:
                print("  Flux : question attendue au tour 1, verdict direct recu")

        if pred.get("type") == "question" and case.get("followup"):
            print("  -> Tour 1 : question posee — envoi du followup patient")
            pred = model.predict_turn2(case["input"], pred, case["followup"])
            turns = 2
            multiturn += 1
            gen_total += 1
            gen_stopped += bool(pred.get("_clean_stop"))

        if pred.get("type") == "question" and not pred.get("question"):
            print("  Champ 'question' manquant")
            missing += 1
        elif pred.get("type") == "final" and not pred.get("analyse"):
            print("  Champ 'analyse' manquant")
            missing += 1

        actual = ("question" if case["expected_type"] == "question"
                  else case.get("expected_urgence", "Inconnue"))
        pred_label = label_of(pred)
        if actual in LABELS and pred_label in LABELS:
            conf[actual][pred_label] += 1

        ok = pred_label == actual
        correct += ok
        if expects_q1 and ok:
            flow_ok += 1
        print(f"  {'OK  ' if ok else 'FAIL'} [{turns}T] attendu={actual:<8} predit={pred_label}")

        if actual == "Haute":
            haute_total += 1
            haute_ok += pred_label == "Haute"

    accuracy     = correct / n
    recall_haute = haute_ok / haute_total if haute_total else 1.0
    stop_rate    = gen_stopped / gen_total if gen_total else 0.0

    print("\n===== MATRICE DE CONFUSION =====")
    print("REEL \\ PRED".ljust(12) + "".join(l.ljust(10) for l in LABELS))
    for a in LABELS:
        print(a.ljust(12) + "".join(str(conf[a][p]).ljust(10) for p in LABELS))

    print("\n===== METRIQUES PAR CLASSE =====")
    print(f"{'Classe':<12} {'Precision':>10} {'Rappel':>10} {'F1':>10}")
    print("-" * 45)
    for l in LABELS:
        tp = conf[l][l]
        fp = sum(conf[a][l] for a in LABELS if a != l)
        fn = sum(conf[l][p] for p in LABELS if p != l)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{l:<12} {prec:>10.2f} {rec:>10.2f} {f1:>10.2f}")

    print("\n===== RESUME =====")
    print(f"Accuracy globale : {accuracy:.2f}   (seuil {THRESHOLDS['accuracy']:.2f})")
    print(f"Rappel Haute     : {recall_haute:.2f}   (seuil {THRESHOLDS['recall_haute']:.2f})")
    print(f"Champs manquants : {missing}")
    print(f"Tours multiples  : {multiturn} exemple(s) evalues en 2 tours")
    if flow_total:
        print(f"Flux Q->verdict  : question au tour 1 : {flow_q1}/{flow_total}"
              f" · verdict final correct : {flow_ok}/{flow_total}")
    print(f"Arret EOS propre : {gen_stopped}/{gen_total} generations ({stop_rate:.0%})")

    # ----- decision CI/CD -----
    eos_min  = float(os.getenv("EOS_STOP_RATE_MIN", "0"))
    blocages = []
    if stop_rate < eos_min:
        blocages.append(f"taux d'arret EOS {stop_rate:.2f} < {eos_min:.2f}")
    if recall_haute < THRESHOLDS["recall_haute"]:
        blocages.append(f"rappel Haute {recall_haute:.2f} < {THRESHOLDS['recall_haute']:.2f}")
    if accuracy < THRESHOLDS["accuracy"]:
        blocages.append(f"accuracy {accuracy:.2f} < {THRESHOLDS['accuracy']:.2f}")
    if missing:
        blocages.append(f"{missing} champ(s) obligatoire(s) manquant(s)")

    for b in blocages:
        print(f"\nBLOCAGE : {b}")
    if blocages:
        print("\nEVALUATION : ECHEC — deploiement bloque.")
        sys.exit(1)
    print("\nEVALUATION : OK — deploiement autorise.")
    sys.exit(0)


if __name__ == "__main__":
    evaluate()
