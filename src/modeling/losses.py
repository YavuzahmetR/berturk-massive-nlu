import torch
import torch.nn as nn
from typing import Optional, Tuple


class JointNLULoss(nn.Module):
    """
    Joint intent + slot loss.
    Class weights ve label smoothing destekler.
    """

    def __init__(
        self,
        slot_pad_idx: int = -100,
        lambda_slot: float = 1.0,
        intent_weights: Optional[torch.Tensor] = None,
        slot_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.slot_pad_idx = slot_pad_idx
        self.lambda_slot = lambda_slot

        self.intent_criterion = nn.CrossEntropyLoss(
            weight=intent_weights,
            label_smoothing=label_smoothing,
        )
        self.slot_criterion = nn.CrossEntropyLoss(
            weight=slot_weights,
            ignore_index=slot_pad_idx,
        )

    def forward(
        self,
        intent_logits: torch.Tensor,
        intent_labels: torch.Tensor,
        slot_logits: torch.Tensor,
        slot_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intent_loss = self.intent_criterion(intent_logits, intent_labels)

        slot_loss = self.slot_criterion(
            slot_logits.view(-1, slot_logits.size(-1)),
            slot_labels.view(-1),
        )

        total = intent_loss + self.lambda_slot * slot_loss
        return total, intent_loss, slot_loss