import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import os

class MSMARCOTripletDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.generate_triplets(raw_data)

    def generate_triplets(self, data):
        triplets = []
        for item in data:
            query = item.get("query", "")
            passages = item.get("passages", {})
            is_selected = passages.get("is_selected", [])
            passage_text = passages.get("passage_text", [])

            # zip the labels and passages together
            combined = list(zip(is_selected, passage_text))

            positives = [text for label, text in combined if label == 1]
            negatives = [text for label, text in combined if label == 0]

            if positives and negatives:
                pos_text = positives[0]
                for neg_text in negatives[:5]:
                    triplets.append((query, pos_text, neg_text))

        print(f"Generated {len(triplets)} triplets.")
        return triplets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, pos, neg = self.samples[idx]
        encoded_query = self.tokenizer(query, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded_pos = self.tokenizer(pos, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        encoded_neg = self.tokenizer(neg, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "query_input_ids": encoded_query["input_ids"].squeeze(0),
            "query_attention_mask": encoded_query["attention_mask"].squeeze(0),
            "pos_input_ids": encoded_pos["input_ids"].squeeze(0),
            "pos_attention_mask": encoded_pos["attention_mask"].squeeze(0),
            "neg_input_ids": encoded_neg["input_ids"].squeeze(0),
            "neg_attention_mask": encoded_neg["attention_mask"].squeeze(0)
        }

def save_dataset():
    print("Loading MS MARCO v1.1 dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    raw_train_data = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Prepare containers
    query_ids, query_masks = [], []
    pos_ids, pos_masks = [], []
    neg_ids, neg_masks = [], []

    print("Generating triplets and tokenizing...")
    for item in raw_train_data:
        query = item.get("query", "")
        passages = item.get("passages", {})
        is_selected = passages.get("is_selected", [])
        passage_text = passages.get("passage_text", [])

        combined = list(zip(is_selected, passage_text))
        positives = [text for label, text in combined if label == 1]
        negatives = [text for label, text in combined if label == 0]

        if positives and negatives:
            pos_text = positives[0]
            for neg_text in negatives[:1]:  # One negative per query
                try:
                    encoded_query = tokenizer(query, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                    encoded_pos = tokenizer(pos_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                    encoded_neg = tokenizer(neg_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

                    query_ids.append(encoded_query["input_ids"][0])
                    query_masks.append(encoded_query["attention_mask"][0])
                    pos_ids.append(encoded_pos["input_ids"][0])
                    pos_masks.append(encoded_pos["attention_mask"][0])
                    neg_ids.append(encoded_neg["input_ids"][0])
                    neg_masks.append(encoded_neg["attention_mask"][0])
                except Exception:
                    continue

    print("Saving tokenized tensors...")
    data = (
        torch.stack(query_ids),
        torch.stack(query_masks),
        torch.stack(pos_ids),
        torch.stack(pos_masks),
        torch.stack(neg_ids),
        torch.stack(neg_masks)
    )
    torch.save(data, "msmarco_triplets_tensor.pt")
    print("✅ Saved to msmarco_triplets_tensor.pt")

if __name__ == "__main__":
    save_dataset()
