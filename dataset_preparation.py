# dataset_preparation.py

import os
import re
import random
import torch
from datasets import load_dataset
from collections import defaultdict

random.seed(42)

def load_ms_marco():
    print("Loading MS MARCO v1.1...")
    dataset = load_dataset("ms_marco", "v1.1")
    return dataset

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def tokenize(text):
    return clean_text(text).split()

def build_vocab(triples, min_freq=2):
    print("Building vocabulary...")
    freq = defaultdict(int)
    for q, p, n in triples:
        for word in tokenize(q) + tokenize(p) + tokenize(n):
            freq[word] += 1

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in freq.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

def encode(text, vocab, max_len=30):
    tokens = tokenize(text)
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_len]]
    if len(token_ids) < max_len:
        token_ids += [vocab["<PAD>"]] * (max_len - len(token_ids))
    return torch.tensor(token_ids)

def create_triples(dataset_split, max_samples=5000):
    print(f"Extracting up to {max_samples} triples...")
    triples = []
    for item in dataset_split:
        query = item["query"]
        positives = item["positive_passages"]
        negatives = item["negative_passages"]
        if positives and negatives:
            triples.append((query, positives[0]["text"], negatives[0]["text"]))
        if len(triples) >= max_samples:
            break
    print(f"Collected {len(triples)} triples.")
    return triples

def process_data(triples, vocab, max_len=30):
    print("Encoding triples...")
    queries, positives, negatives = [], [], []
    for q, p, n in triples:
        queries.append(encode(q, vocab, max_len))
        positives.append(encode(p, vocab, max_len))
        negatives.append(encode(n, vocab, max_len))
    return torch.stack(queries), torch.stack(positives), torch.stack(negatives)

def save_data(q, p, n, vocab, out_dir="processed_data"):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"query": q, "positive": p, "negative": n}, os.path.join(out_dir, "triples.pt"))
    torch.save(vocab, os.path.join(out_dir, "vocab.pt"))
    print(f"Saved data to {out_dir}/")

if __name__ == "__main__":
    dataset = load_ms_marco()
    triples = create_triples(dataset["train"], max_samples=5000)
    vocab = build_vocab(triples)
    q, p, n = process_data(triples, vocab)
    save_data(q, p, n, vocab)
