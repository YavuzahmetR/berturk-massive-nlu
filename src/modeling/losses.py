import torch
import torch.nn as nn
from typing import Tuple

class JointNLULoss(nn.Module):
    """
    Multi-task loss function combining Intent Classification and Slot Filling.
    """
    def __init__(self, slot_pad_idx: int = -100, lambda_slot: float = 1.0) -> None:
        super().__init__()
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=slot_pad_idx)
        self.lambda_slot = lambda_slot

    def forward(
        self, 
        intent_logits: torch.Tensor, 
        intent_labels: torch.Tensor, 
        slot_logits: torch.Tensor, 
        slot_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intent_loss = self.intent_loss_fn(intent_logits, intent_labels)

        num_slots = slot_logits.shape[-1]
        flat_slot_logits = slot_logits.view(-1, num_slots)
        flat_slot_labels = slot_labels.view(-1)

        slot_loss = self.slot_loss_fn(flat_slot_logits, flat_slot_labels)
        total_loss = intent_loss + (self.lambda_slot * slot_loss)

        return total_loss, intent_loss, slot_loss
