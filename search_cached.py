import torch
import pickle
from transformers import AutoTokenizer
from two_tower import TwoTowerModel
from torch.nn.functional import cosine_similarity

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128
TOP_K = 5

# --- Load cached vectors & texts ---
doc_vectors = torch.load("doc_vectors.pt")  # shape: [N_docs, hidden_dim]
with open("doc_texts.pkl", "rb") as f:
    doc_texts = pickle.load(f)

# --- Load tokenizer and model ---
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

# --- Fast retrieval function ---
def search(query_text, top_k=5):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(query_text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        query_vec = model.encode_query(input_ids, attention_mask).cpu()
        sims = cosine_similarity(query_vec, doc_vectors)  # shape: [N_docs]

        top_indices = sims.argsort(descending=True)[:top_k]

        print(f"\n🔍 Query: {query_text}")
        print("Top {top_k} results:")
        for rank, idx in enumerate(top_indices):
            score = sims[idx].item()
            passage = doc_texts[idx]
            print(f"{rank+1}. [Score: {score:.4f}] {passage[:100]}...")

# === Accept any query input from user
query = input("Enter your query: ")
search(query, top_k=5)

# --- Try it!
search("what is results-based accountability")
