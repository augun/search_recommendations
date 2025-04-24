import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from two_tower import TwoTowerModel
import os

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
DOC_CACHE_PATH = "cached_doc_vectors.pt"
TOP_K = 5

# --- Load tokenizer and GloVe embeddings ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_tensor = torch.load("glove_embeddings.pt")
embedding_layer = torch.nn.Embedding.from_pretrained(embedding_tensor, freeze=True)

# --- Initialize model ---
model = TwoTowerModel(embedding=embedding_layer, hidden_dim=128)
model.load_state_dict(torch.load("two_tower_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Sample document store (you can replace this with full corpus) ---
from datasets import load_dataset
print("Loading documents from MS MARCO...")
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train[:1000]")

sample_passages = []
for sample in dataset:
    passages = sample["passages"]["passage_text"]
    sample_passages.extend([p for p in passages if isinstance(p, str)])

# --- Cache document encodings ---
if os.path.exists(DOC_CACHE_PATH):
    print("🔁 Loading cached document vectors...")
    cached = torch.load(DOC_CACHE_PATH)
    doc_texts = cached["texts"]
    doc_embeddings = cached["embeddings"].to(DEVICE)
else:
    print("⚙️ Encoding documents...")
    doc_texts = sample_passages # You can increase this
    doc_tokens = tokenizer(doc_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
    doc_ids = doc_tokens["input_ids"].to(DEVICE)
    doc_mask = doc_tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():
        doc_embeddings = model.encode_document(doc_ids, doc_mask)

    torch.save({"texts": doc_texts, "embeddings": doc_embeddings.cpu()}, DOC_CACHE_PATH)
    print(f"✅ Cached {len(doc_texts)} documents to {DOC_CACHE_PATH}")

# --- Inference loop ---
def search_loop():
    while True:
        query = input("\n🔍 Enter your query (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        query_tokens = tokenizer(query, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
        q_ids = query_tokens["input_ids"].to(DEVICE)
        q_mask = query_tokens["attention_mask"].to(DEVICE)

        with torch.no_grad():
            query_vec = model.encode_query(q_ids, q_mask)
            sims = F.cosine_similarity(query_vec, doc_embeddings)
            top_k = sims.topk(TOP_K)

        print(f"\n🔝 Top {TOP_K} results:")
        for i, idx in enumerate(top_k.indices.tolist()):
            print(f"{i+1}. [Score: {top_k.values[i]:.4f}] {doc_texts[idx][:100]}...")

if __name__ == "__main__":
    search_loop()
