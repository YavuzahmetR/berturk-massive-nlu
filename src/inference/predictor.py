import os
import json
from typing import List, Dict, Any, Tuple

import torch
import yaml
from transformers import AutoTokenizer

from src.modeling.joint_nlu_model import JointTurkishNLUModel


def decode_bio_entities(
    slot_preds: List[int],
    offsets: List[Tuple[int, int]],
    attention: List[int],
    id_to_slot: Dict[int, str],
    text: str,
) -> List[Dict[str, Any]]:
    """
    BIO token-level tahminlerini entity span'lerine çevirir.
    Alt-token birleştirme: iki token arasında boşluk yoksa ve offset'ler
    bitişikse aynı kelimenin parçası sayılır (örn. 'bok' + '##s' -> 'boks').
    """
    entities: List[Dict[str, Any]] = []
    current: Dict[str, Any] = None

    for pred_id, offset, attn in zip(slot_preds, offsets, attention):
        if attn == 0 or offset == (0, 0):
            continue

        label = id_to_slot.get(pred_id, "O")
        start, end = offset
        token_text = text[start:end]

        # --- Alt-token birleştirme ---
        # Önceki entity'nin bittiği yer bu token'ın başladığı yerle aynıysa
        # ve arada boşluk yoksa, bu token aynı kelimenin parçasıdır.
        if (
            current is not None
            and start == current["end"]
            and token_text
            and not token_text[0].isspace()
        ):
            if label.startswith("B-"):
                new_type = label[2:]
                if new_type != current["type"]:
                    # Farklı tipte yeni entity → öncekini kapat, yenisini başlat
                    entities.append(current)
                    current = {"type": new_type, "start": start, "end": end}
                    continue
            # Aynı entity'nin devamı → sadece end'i uzat
            current["end"] = end
            continue

        if label == "O":
            if current is not None:
                entities.append(current)
                current = None
            continue

        prefix, _, ent_type = label.partition("-")

        if prefix == "B":
            if current is not None:
                entities.append(current)
            current = {"type": ent_type, "start": start, "end": end}
        elif prefix == "I":
            if current is not None and current["type"] == ent_type:
                current["end"] = end
            else:
                if current is not None:
                    entities.append(current)
                current = {"type": ent_type, "start": start, "end": end}

    if current is not None:
        entities.append(current)

    for ent in entities:
        ent["text"] = text[ent["start"]:ent["end"]]

    return entities


class NLUPredictor:
    """
    Eğitilmiş Joint NLU modelini bir kez yükler ve predict / predict_batch sunar.
    FastAPI gibi çoklu istek alan ortamlarda güvenle paylaşılabilir.
    """

    def __init__(
        self,
        config_path: str = "configs/joint_bert.yaml",
        checkpoint_path: str = "checkpoints/best_joint_nlu_model.pt",
        device: str = None,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        with open(
            os.path.join("data", "processed", "intent_to_id.json"),
            "r",
            encoding="utf-8",
        ) as f:
            self.intent_to_id = json.load(f)
        with open(
            os.path.join("data", "processed", "slot_to_id.json"),
            "r",
            encoding="utf-8",
        ) as f:
            self.slot_to_id = json.load(f)

        self.id_to_intent = {v: k for k, v in self.intent_to_id.items()}
        self.id_to_slot = {v: k for k, v in self.slot_to_id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg["model"]["model_name"])
        self.max_length = self.cfg["model"]["max_length"]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint_path}")

        self.model = JointTurkishNLUModel(
            model_name=self.cfg["model"]["model_name"],
            num_intents=len(self.intent_to_id),
            num_slots=len(self.slot_to_id),
            dropout_rate=self.cfg["model"]["dropout_rate"],
        )
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, Any]:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long).to(self.device)

        intent_logits, slot_logits = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        intent_probs = torch.softmax(intent_logits, dim=-1)
        intent_ids = torch.argmax(intent_logits, dim=-1).cpu().tolist()
        intent_confs = intent_probs.max(dim=-1).values.cpu().tolist()

        slot_preds = torch.argmax(slot_logits, dim=-1).cpu().tolist()

        results = []
        for i, text in enumerate(texts):
            entities = decode_bio_entities(
                slot_preds[i],
                encodings["offset_mapping"][i],
                encodings["attention_mask"][i],
                self.id_to_slot,
                text,
            )
            results.append(
                {
                    "text": text,
                    "intent": self.id_to_intent[intent_ids[i]],
                    "confidence": float(intent_confs[i]),
                    "entities": entities,
                }
            )
        return results