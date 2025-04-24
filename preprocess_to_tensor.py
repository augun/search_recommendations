import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np

# --- Config ---
MAX_LENGTH = 128
SAVE_PATH = "msmarco_triplets_tensor.pt"
NUM_NEGATIVES = 3

# --- Load tokenizer and embeddings ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_dim = 300

# Optional: Use pretrained GloVe or word2vec embeddings here
# For simplicity, we're initializing random embeddings here (replace this with real pretrained ones if available)
vocab_size = tokenizer.vocab_size
pretrained_embeddings = torch.randn(vocab_size, embedding_dim)  # Replace with real embeddings

# --- Load dataset ---
dataset = load_dataset("microsoft/ms_marco", "v1.1")

# Prepare containers
query_ids, query_masks = [], []
pos_ids, pos_masks = [], []
neg_ids, neg_masks = [], []

print("Processing dataset and tokenizing triplets...")
for sample in tqdm(dataset["train"]):
    passages = sample["passages"]["passage_text"]
    labels = sample["passages"]["is_selected"]

    if 1 not in labels:
        continue

    try:
        query_text = sample["query"]
        pos_text = passages[labels.index(1)]
        neg_texts = [t for t, sel in zip(passages, labels) if sel == 0]

        if not neg_texts:
            continue

        neg_samples = random.sample(neg_texts, min(len(neg_texts), NUM_NEGATIVES))

        for neg_text in neg_samples:
            q = tokenizer(query_text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
            p = tokenizer(pos_text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
            n = tokenizer(neg_text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")

            query_ids.append(q["input_ids"][0])
            query_masks.append(q["attention_mask"][0])
            pos_ids.append(p["input_ids"][0])
            pos_masks.append(p["attention_mask"][0])
            neg_ids.append(n["input_ids"][0])
            neg_masks.append(n["attention_mask"][0])

    except Exception:
        continue

print("Saving tokenized tensors...")
data = (
    torch.stack(query_ids),
    torch.stack(query_masks),
    torch.stack(pos_ids),
    torch.stack(pos_masks),
    torch.stack(neg_ids),
    torch.stack(neg_masks),
    pretrained_embeddings  # save this for model loading
)
torch.save(data, SAVE_PATH)
print(f"✅ Saved preprocessed triplets with negatives to {SAVE_PATH}")