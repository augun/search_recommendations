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
    # Load dataset
    print("Loading MS MARCO v1.1 dataset...")
    dataset = load_dataset("microsoft/ms_marco", "v1.1")
    raw_train_data = dataset["train"]

    # print("First sample:", raw_train_data[0])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Prepare dataset
    triplet_dataset = MSMARCOTripletDataset(raw_train_data, tokenizer)

    # Save dataset as a torch file
    output_path = "msmarco_triplets.pt"
    torch.save(triplet_dataset, output_path)
    print(f"Saved preprocessed dataset to {output_path}")

if __name__ == "__main__":
    save_dataset()
