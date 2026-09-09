import re
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

class MassiveAnnotationParser:
    """
        Parses MASSIVE-style annotated utterances.
    
        Example:
            "yarın [time : sabah yediye] alarm kur"
    
        Output:
            cleaned_text:
                "yarın sabah yediye alarm kur"
    
            spans:
                [
                    {
                        "type": "time",
                        "text": "sabah yediye",
                        "start": 5,
                        "end": 17,
                    }
                ]
    
        Important:
            Span indices refer to the CLEANED text, not the original
            annotated string. This is critical for tokenizer alignment.
        """

    def __init__(self) -> None:
        # Captures:
        #   [slot_type : slot_text]
        #
        # Example:
        #   [time : sabah yediye]
        self.pattern = re.compile(
           r"\[([a-zA-Z0-9_\-]+)\s*:\s*([^\]]+)\]"
        )

    def parse(self, annot_utt: str) -> Tuple[str,List[Dict[str, Any]]] : 
        """
        Converts an annotated utterance into:
        1. cleaned text
        2. character spans on that cleaned text
        """

        spans: List[Dict[str, Any]] = []

        # We build the cleaned text piece-by-piece rather than modifying
        # the original string inside a loop. This keeps character offsets
        # consistent even when there are multiple slots.

        cleaned_parts : List[str] = []

        cursor = 0
        cleaned_cursor = 0

        for match in self.pattern.finditer(annot_utt):
            # Add normal text before the annotation.
            prefix = annot_utt[cursor:match.start()]
            cleaned_parts.append(prefix)
            cleaned_cursor += len(prefix)

            slot_type = match.group(1).strip()
            slot_text = match.group(2).strip()

            # The slot text is inserted into the cleaned sentence
            slot_start = cleaned_cursor
            cleaned_parts.append(slot_text)
            cleaned_cursor += len(slot_text)
            slot_end = cleaned_cursor

            spans.append(
                {
                    "type" : slot_type,
                    "text" : slot_text,
                    "start" : slot_start,
                    "end" : slot_end
                }
            )

            cursor = match.end()
        # Add any remaining text after the final annotation.

        suffix = annot_utt[cursor:]
        cleaned_parts.append(suffix)

        cleaned_text = "".join(cleaned_parts)

        return cleaned_text, spans

class TokenAlignmentEncoder:
    """
    Converts character-level slot spans into token-level BIO labels.
    
        Example:
    
            Text:
                "yarın sabah yediye alarm kur"
    
            Slot:
                "sabah yediye" -> time
    
            Possible token labels:
                yarın  -> O
                sabah  -> B-time
                yedi   -> I-time
                ##ye   -> I-time
                alarm  -> O
                kur    -> O
    
        Special tokens such as [CLS] and [SEP] receive -100 so that
        CrossEntropyLoss can ignore them.
    """

    def __init__(self, slot_to_id: Dict[str, int]) -> None:
        if "O" not in slot_to_id:
            raise ValueError("slot_to_id must contain the 'O' label.")
        self.slot_to_id = slot_to_id

    def align_slots_to_tokens(self, spans: List[Dict[str, Any]], offset_mapping: List[Tuple[int,int]]) -> List[int]:
        """
        Maps cleaned-text character spans to tokenizer offsets.
        
        offset_mapping example:
        
                token       offset
                -------------------
                [CLS]       (0, 0)
                yarın       (0, 4)
                sabah       (5, 10)
                yedi        (11, 15)
                ##ye        (15, 17)
                alarm       (18, 23)
                kur         (24, 27)
                [SEP]       (0, 0)
        """
        #Start every real token as "O".
        labels = [self.slot_to_id["O"]] * len(offset_mapping)
        # Special tokens have no real character span.
        # -100 tells CrossEntropyLoss to ignore these positions.

        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == 0 and token_end == 0:
                labels[token_idx] = -100
        for span in spans:
            slot_type = span["type"]
            span_start = span["start"]
            span_end = span["end"]

            b_label = f"B-{slot_type}"
            i_label = f"I-{slot_type}"

            if b_label not in self.slot_to_id:
                raise KeyError(f"Missing BIO label in slot_to_id:{b_label}")

            if i_label not in self.slot_to_id:
                raise KeyError(f"Missing BIO label in slot_to_id:{i_label}")

            first_token_found = False

            for token_idx,(token_start, token_end) in enumerate(offset_mapping):
                #Ignore special tokens
                if token_start == 0 and token_end == 0 :
                    continue
                # A token belongs to the slot when its character span
                # overlaps the slot span.
                # This is safer than requiring:
                #token_start >= span_start
                # token_end <= span_end
                # because subword tokenization can create boundary cases

                overlaps = (
                    token_start < span_end and token_end > span_start
                )

                if not overlaps:
                    continue
                if not first_token_found:
                    labels[token_idx] = self.slot_to_id[b_label]
                    first_token_found = True
                else:
                    labels[token_idx] = self.slot_to_id[i_label]

        return labels
                
class JointTurkishNLUModel(nn.Module):
    """
        Joint Intent Classification + Slot Filling model.
    
        Architecture:
    
                      Input
                        |
                     BERTurk
                        |
              +---------+---------+
              |                   |
           CLS vector        Token vectors
              |                   |
         Intent Head          Slot Head
              |                   |
           Intent             BIO labels
    """

    def __init__(self, model_name: str, num_intents : int, num_slots : int, dropout_rate : float = 0.1) -> None:
        super().__init__()

        # Shared Transformer encoder.
        self.encoder = AutoModel.from_pretrained(model_name)

        self.dropout = nn.Dropout(dropout_rate)

        hidden_size = self.encoder.config.hidden_size

        # Intent:
        # One prediction for the whole sentence.

        self.intent_classifier = nn.Linear(
            hidden_size,
            num_intents
        )

        # Slots:
        # One prediction for every token.

        self.slot_classifier = nn.Linear(
            hidden_size,
            num_slots
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one forward pass.
        
        Returns:
            intent_logits:
            Shape -> (batch_size, num_intents)
        
            slot_logits:
            Shape -> (batch_size, seq_len, num_slots)
        """

        outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        # Contextual representation for every token.
        # Shape:
        #   (batch_size, seq_len, hidden_size)

        sequence_output = outputs.last_hidden_state
        # We use the first token ([CLS]) as the sentence-level
         # representation for intent classification.
        # Shape:
         #   (batch_size, hidden_size)

        pooled_output = sequence_output[:, 0, :]
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # One intent prediction for the entire sentence.

        intent_logits = self.intent_classifier(
            pooled_output
        )

        # One slot prediction for every token.
        slot_logits = self.slot_classifier(
            sequence_output
        )

        return intent_logits, slot_logits
        










        