"""Original CLI decoding, intentionally separate from API subtoken decoding."""

import os
import json
import torch
from transformers import AutoTokenizer



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
