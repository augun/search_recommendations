import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from two_tower import TwoTowerModel
from dataset_preparation import MSMARCOTripletDataset
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.nn.functional import cosine_similarity

# --- Config ---
BATCH_SIZE = 32
MAX_LENGTH = 128
HIDDEN_DIM = 128
EMBEDDING_DIM = 300
LR = 1e-4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load dataset ---
print("Loading MS MARCO dataset...")
dataset = load_dataset("microsoft/ms_marco", "v1.1")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_data = dataset["train"].select(range(1000))  # small subset for faster testing
train_dataset = MSMARCOTripletDataset(train_data, tokenizer, max_length=MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Initialize model ---
print("Building model...")
vocab_size = tokenizer.vocab_size
model = TwoTowerModel(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM
)
model.to(DEVICE)

# --- Loss and optimizer ---
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training ---
print("Training started on", DEVICE)
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        q_ids = batch["query_input_ids"].to(DEVICE)
        q_mask = batch["query_attention_mask"].to(DEVICE)
        pos_ids = batch["pos_input_ids"].to(DEVICE)
        pos_mask = batch["pos_attention_mask"].to(DEVICE)
        neg_ids = batch["neg_input_ids"].to(DEVICE)
        neg_mask = batch["neg_attention_mask"].to(DEVICE)

        q_vec, pos_vec, neg_vec = model(
            q_ids, q_mask,
            pos_ids, pos_mask,
            neg_ids, neg_mask
        )

        loss = criterion(q_vec, pos_vec, neg_vec)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# --- Save the model ---
torch.save(model.state_dict(), "two_tower_model.pt")
print("Model saved to two_tower_model.pt ✅")

# --- Simple inference: ranking docs by cosine similarity ---
def rank_documents(query_text, doc_texts, model, tokenizer):
    model.eval()
    with torch.no_grad():
        query_tokens = tokenizer(query_text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
        doc_tokens = tokenizer(doc_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)

        q_ids = query_tokens["input_ids"].to(DEVICE)
        q_mask = query_tokens["attention_mask"].to(DEVICE)
        d_ids = doc_tokens["input_ids"].to(DEVICE)
        d_mask = doc_tokens["attention_mask"].to(DEVICE)

        query_vec = model.encode_query(q_ids, q_mask)
        doc_vecs = model.encode_document(d_ids, d_mask)

        sims = cosine_similarity(query_vec.unsqueeze(0), doc_vecs)
        ranked = sorted(zip(doc_texts, sims.squeeze(0).tolist()), key=lambda x: x[1], reverse=True)

        print("\n🔍 Query:", query_text)
        print("Top matches:")
        for i, (doc, score) in enumerate(ranked[:5]):
            # Flatten anything nested into a plain string
            doc_text = doc
            if isinstance(doc, list):
                doc_text = " ".join(doc)  # join multiple strings if it's a list
            elif not isinstance(doc, str):
                doc_text = str(doc)  # fallback just in case

            print(f"{i+1}. [Score: {score:.4f}] {doc_text[:100]}...")


# --- Use passages from a single item where passage_text is a list of strings ---
sample_item = train_data[0]
example_query = sample_item["query"]

# Get 10 document strings directly from the same sample
all_passages = sample_item["passages"]["passage_text"]
example_docs = [text for text in all_passages if isinstance(text, str)][:10]

# Confirm doc types
print("\n🔎 Sample query:", example_query)
print("Example doc types:", [type(doc) for doc in example_docs])
print("Example doc preview:", example_docs[0][:100])

# Run inference
rank_documents(example_query, example_docs, model, tokenizer)


