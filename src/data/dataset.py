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
        self.raw_examples = raw_examples
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

        if not annot_utt or not intent_name:
            raise ValueError(f"Malformed dataset row at index {idx}. Check raw schema bounds.")

        cleaned_text, spans = self.parser.parse(annot_utt)

        encoding = self.tokenizer(
        cleaned_text,
        return_offsets_mapping=True,
        )

        print("\n[DEBUG] TOKENIZER OUTPUT KEYS:")
        print(encoding.keys())
        print("[DEBUG] TOKENIZER TYPE:", type(self.tokenizer))
        print("[DEBUG] TOKENIZER IS FAST:", self.tokenizer.is_fast)

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"]

        bio_labels = self.encoder.align_slots_to_tokens(spans, offset_mapping)

        if intent_name not in self.intent_to_id:
            raise KeyError(f"Intent target token '{intent_name}' is not registered in lookups.")

        intent_id = self.intent_to_id[intent_name]   

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "slot_labels": bio_labels,
            "intent_label": intent_id               
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
        "intent_label": torch.tensor(intent_labels, dtype=torch.long)
    }