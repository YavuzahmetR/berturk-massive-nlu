import re
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Any, Dict, List, Tuple

class MassiveAnnotationParser:
    """
    Parses MASSIVE-style annotated utterances.
    """
    def __init__(self) -> None:
        self.pattern = re.compile(
           r"\[([a-zA-Z0-9_\-]+)\s*:\s*([^\]]+)\]"
        )

    def parse(self, annot_utt: str) -> Tuple[str, List[Dict[str, Any]]]: 
        spans: List[Dict[str, Any]] = []
        cleaned_parts: List[str] = []
        cursor = 0
        cleaned_cursor = 0

        for match in self.pattern.finditer(annot_utt):
            prefix = annot_utt[cursor:match.start()]
            cleaned_parts.append(prefix)
            cleaned_cursor += len(prefix)

            slot_type = match.group(1).strip()
            slot_text = match.group(2).strip()

            slot_start = cleaned_cursor
            cleaned_parts.append(slot_text)
            cleaned_cursor += len(slot_text)
            slot_end = cleaned_cursor

            spans.append({
                "type": slot_type,
                "text": slot_text,
                "start": slot_start,
                "end": slot_end
            })
            cursor = match.end()

        suffix = annot_utt[cursor:]
        cleaned_parts.append(suffix)
        cleaned_text = "".join(cleaned_parts)

        return cleaned_text, spans

class TokenAlignmentEncoder:
    """
    Converts character-level slot spans into token-level BIO labels.
    """
    def __init__(self, slot_to_id: Dict[str, int]) -> None:
        if "O" not in slot_to_id:
            raise ValueError("slot_to_id must contain the 'O' label.")
        self.slot_to_id = slot_to_id

    def align_slots_to_tokens(self, spans: List[Dict[str, Any]], offset_mapping: List[Tuple[int, int]]) -> List[int]:
        labels = [self.slot_to_id["O"]] * len(offset_mapping)

        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == 0 and token_end == 0:
                labels[token_idx] = -100

        for span in spans:
            slot_type = span["type"]
            span_start = span["start"]
            span_end = span["end"]

            b_label = f"B-{slot_type}"
            i_label = f"I-{slot_type}"

            if b_label not in self.slot_to_id or i_label not in self.slot_to_id:
                raise KeyError(f"Missing BIO label in slot_to_id mapping for {slot_type}")

            first_token_found = False

            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == 0 and token_end == 0:
                    continue

                overlaps = (token_start < span_end and token_end > span_start)

                if not overlaps:
                    continue

                if not first_token_found:
                    labels[token_idx] = self.slot_to_id[b_label]
                    first_token_found = True
                else:
                    if labels[token_idx] == self.slot_to_id["O"]:
                        labels[token_idx] = self.slot_to_id[i_label]

        return labels
                
class JointTurkishNLUModel(nn.Module):
    """
    Joint Intent Classification + Slot Filling model using BERTurk.
    """
    def __init__(self, model_name: str, num_intents: int, num_slots: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        hidden_size = self.encoder.config.hidden_size

        self.intent_classifier = nn.Linear(hidden_size, num_intents)
        self.slot_classifier = nn.Linear(hidden_size, num_slots)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0, :]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits