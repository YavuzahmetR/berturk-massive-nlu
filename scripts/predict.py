import os
import json
import sys
import torch
import yaml
from transformers import AutoTokenizer

from src.modeling.joint_nlu_model import (
    JointTurkishNLUModel,
    TokenAlignmentEncoder,
)


def load_artifacts(cfg):
    with open(os.path.join("data", "processed", "intent_to_id.json"), "r", encoding="utf-8") as f:
        intent_to_id = json.load(f)
    with open(os.path.join("data", "processed", "slot_to_id.json"), "r", encoding="utf-8") as f:
        slot_to_id = json.load(f)
    id_to_intent = {v: k for k, v in intent_to_id.items()}
    id_to_slot = {v: k for k, v in slot_to_id.items()}
    return intent_to_id, slot_to_id, id_to_intent, id_to_slot


def decode_bio_entities(slot_preds, offsets, attention, id_to_slot, text):
    """BIO token tahminlerini entity span'lerine dönüştürür."""
    entities = []
    current = None

    for pred_id, offset, attn in zip(slot_preds, offsets, attention):
        if attn == 0 or offset == (0, 0):
            continue

        label = id_to_slot.get(pred_id, "O")

        if label == "O":
            if current:
                entities.append(current)
                current = None
            continue

        prefix, _, ent_type = label.partition("-")

        if prefix == "B":
            if current:
                entities.append(current)
            current = {"type": ent_type, "start": offset[0], "end": offset[1]}
        elif prefix == "I":
            if current and current["type"] == ent_type:
                current["end"] = offset[1]
            else:
                if current:
                    entities.append(current)
                current = {"type": ent_type, "start": offset[0], "end": offset[1]}

    if current:
        entities.append(current)

    for ent in entities:
        ent["text"] = text[ent["start"]:ent["end"]]

    return entities


def predict(text, model, tokenizer, id_to_intent, id_to_slot, device, max_length):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long).to(device)
    attention_mask = torch.tensor([encoding["attention_mask"]], dtype=torch.long).to(device)

    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids=input_ids, attention_mask=attention_mask)

    intent_id = torch.argmax(intent_logits, dim=-1).item()
    intent_name = id_to_intent[intent_id]
    intent_conf = torch.softmax(intent_logits, dim=-1)[0, intent_id].item()

    slot_preds = torch.argmax(slot_logits, dim=-1)[0].cpu().tolist()
    entities = decode_bio_entities(
        slot_preds,
        encoding["offset_mapping"],
        encoding["attention_mask"],
        id_to_slot,
        text,
    )

    return intent_name, intent_conf, entities


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
        print("🏷️  Slots  :")
        for ent in entities:
            print(f"     • {ent['type']:<20s} -> \"{ent['text']}\"")
    else:
        print("🏷️  Slots  : (yok)")


if __name__ == "__main__":
    main()