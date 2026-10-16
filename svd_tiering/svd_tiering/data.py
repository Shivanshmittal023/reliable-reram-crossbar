"""GLUE SST-2 loading and evaluation."""
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def sst2_loader(tokenizer, split: str, n: int | None, batch_size: int,
                shuffle: bool, seed: int = 0, max_len: int = 128) -> DataLoader:
    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    ds = ds.map(lambda b: tokenizer(b["sentence"], truncation=True, max_length=max_len),
                batched=True, remove_columns=["sentence", "idx"])
    ds = ds.rename_column("label", "labels")
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      collate_fn=DataCollatorWithPadding(tokenizer))


@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        logits = model(**batch).logits
        correct += (logits.argmax(-1) == batch["labels"]).sum().item()
        total += batch["labels"].numel()
    return correct / total
