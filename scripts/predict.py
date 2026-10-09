"""Terminal inference and output formatting; predictions are unchanged."""

import os
import sys

import torch
import yaml
from transformers import AutoTokenizer

from src.inference.cli_predictor import decode_bio_entities, load_artifacts, predict
from src.modeling.joint_nlu_model import JointTurkishNLUModel


def main():
    print("🔮 [INFERENCE DEMO] Turkish Joint NLU — Interactive Mode\n")

    # Config
    config_path = os.path.join("configs", "joint_bert.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    intent_to_id, slot_to_id, id_to_intent, id_to_slot = load_artifacts(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Computing Hardware Engine Target: {device}")

    model = JointTurkishNLUModel(
        model_name=cfg["model"]["model_name"],
        num_intents=len(intent_to_id),
        num_slots=len(slot_to_id),
        dropout_rate=cfg["model"]["dropout_rate"],
    )

    checkpoint_path = "best_joint_nlu_model.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[CRITICAL] Checkpoint missing: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    max_length = cfg["model"]["max_length"]

    # CLI mode: pass sentence as argument
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        run_single(sentence, model, tokenizer, id_to_intent, id_to_slot, device, max_length)
        return

    # Interactive mode
    print("[INFO] Cümle yaz, çıkmak için 'q' / 'exit' / Ctrl+C\n")
    print("-" * 60)
    while True:
        try:
            text = input("📝 Cümle: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Çıkılıyor.")
            break

        if text.lower() in {"q", "quit", "exit"}:
            print("[INFO] Çıkılıyor.")
            break
        if not text:
            continue

        print("-" * 60)
        run_single(text, model, tokenizer, id_to_intent, id_to_slot, device, max_length)
        print("-" * 60)


def run_single(text, model, tokenizer, id_to_intent, id_to_slot, device, max_length):
    intent, conf, entities = predict(
        text, model, tokenizer, id_to_intent, id_to_slot, device, max_length
    )

    print(f"🎯 Intent : {intent}  (confidence: {conf * 100:.2f}%)")

    if entities:
        print(" Slots  :")
        for ent in entities:
            print(f"     • {ent['type']:<20s} -> \"{ent['text']}\"")
    else:
        print(" Slots  : (none)")


if __name__ == "__main__":
    main()
