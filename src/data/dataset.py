import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any


class MassiveJointNLUDataset(Dataset):
    """
    Leakage-free PyTorch Dataset for joint intent and slot training.
    """

    def __init__(
            self,
            raw_examples: List[Dict[str, Any]],
            parser: Any,
            encoder: Any,
            tokenizer: Any,
            intent_to_id: Dict[str, int],
            max_length: int = 64
    ) -> None:
        # Boş / eksik satırları en baştan temizle (0 geçerli bir ID olabilir!)
        def gecerli_mi(ex: Dict[str, Any]) -> bool:
            annot = ex.get("annot_utt")
            intent = ex.get("intent")
            if annot is None or intent is None:
                return False
            if str(annot).strip() == "":
                return False
            if str(intent).strip() == "":
                return False
            return True

        self.raw_examples = [ex for ex in raw_examples if gecerli_mi(ex)]
        print(f"[DATA] {len(raw_examples)} satırdan {len(self.raw_examples)} tanesi geçerli.")

        self.parser = parser
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.intent_to_id = intent_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.raw_examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.raw_examples[idx]

        annot_utt = example["annot_utt"]
        intent_name = example["intent"]

        cleaned_text, spans = self.parser.parse(annot_utt)

        encoding = self.tokenizer(
            cleaned_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"]

        bio_labels = self.encoder.align_slots_to_tokens(spans, offset_mapping)

        assert len(input_ids) == self.max_length, \
            f"Beklenmeyen token uzunluğu {len(input_ids)} != {self.max_length}"
        assert len(bio_labels) == self.max_length, \
            f"Beklenmeyen slot etiket uzunluğu {len(bio_labels)} != {self.max_length}"

        intent_str = str(intent_name).strip()
        if intent_str.isdigit():
            intent_id = int(intent_str)
        else:
            if intent_str not in self.intent_to_id:
                raise KeyError(f"Intent '{intent_str}' intent_to_id içinde yok.")
            intent_id = self.intent_to_id[intent_str]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slot_labels": bio_labels,
            "intent_label": intent_id,
        }


def joint_nlu_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    slot_labels = [item["slot_labels"] for item in batch]
    intent_labels = [item["intent_label"] for item in batch]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
        "intent_label": torch.tensor(intent_labels, dtype=torch.long),
    }