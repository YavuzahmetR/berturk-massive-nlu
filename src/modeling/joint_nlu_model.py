"""BERTurk encoder with separate intent and slot classifiers."""

from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

# Preserve the public imports used by old scripts, notebooks and tests.
from src.data.annotations import MassiveAnnotationParser, TokenAlignmentEncoder


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
