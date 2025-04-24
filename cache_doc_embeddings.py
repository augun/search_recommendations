import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from two_tower import TwoTowerModel
from datasets import load_dataset
import pickle
from tqdm import tqdm

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LENGTH = 128

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

model = TwoTowerModel(
    vocab_size=vocab_size,
    embedding_dim=300,
    hidden_dim=128
)
model.load_state_dict(torch.load("two_tower_model.pt"))
model.to(DEVICE)
model.eval()

# Load dataset
dataset = load_dataset("microsoft/ms_marco", "v1.1")
doc_items = dataset["train"] # adjust this to cache more

# --- Pre-encode document embeddings ---
encoded_vectors = []
raw_passages = []

with torch.no_grad():
    for item in tqdm(doc_items):
        passages = item["passages"]["passage_text"]
        for text in passages:
            if not isinstance(text, str):
                continue

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            # Encode using document tower
            vec = model.encode_document(input_ids, attention_mask)
            encoded_vectors.append(vec.squeeze().cpu())
            raw_passages.append(text)

# --- Save to disk
torch.save(torch.stack(encoded_vectors), "doc_vectors.pt")
with open("doc_texts.pkl", "wb") as f:
    pickle.dump(raw_passages, f)

print(f"\n✅ Cached {len(encoded_vectors)} document vectors.")
