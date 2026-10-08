"""Read the local JSONL snapshots without altering their records."""

import json


def load_local_jsonl(file_path: str) -> list:
    examples = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))
    return examples
