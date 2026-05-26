"""SST-2 (GLUE) data loading for the NLP LoRA experiments."""

from __future__ import annotations

from typing import Optional, Tuple


def load_sst2(
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    cache_dir: Optional[str] = None,
):
    """Tokenise SST-2 and return ``(train_dataset, validation_dataset, tokenizer)``.

    The SST-2 test split has hidden labels, so we follow the original notebook
    and evaluate on the GLUE ``validation`` split, which has 872 labelled
    examples.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    raw = load_dataset("glue", "sst2", cache_dir=cache_dir)

    def tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    return tokenized["train"], tokenized["validation"], tokenizer
