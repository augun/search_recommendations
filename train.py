import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
from two_tower import TwoTowerModel
import torch.nn as nn
from torch.nn.functional import cosine_similarity

# --- Config ---
BATCH_SIZE = 32
HIDDEN_DIM = 128
EMBEDDING_DIM = 300
LR = 1e-4
EPOCHS = 20
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load train and val datasets ---
print("Loading train and validation data with GloVe embeddings...")
train_data = torch.load("train_tensor.pt")
val_data = torch.load("val_tensor.pt")

train_tensors = train_data[:6]
val_tensors = val_data[:6]
pretrained_embeddings = train_data[6]  # same as val_data[6]

train_dataset = TensorDataset(*train_tensors)
val_dataset = TensorDataset(*val_tensors)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Initialize model ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_layer = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
model = TwoTowerModel(embedding=embedding_layer, hidden_dim=HIDDEN_DIM)
model.to(DEVICE)

# --- Loss and optimizer ---
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training with validation and early stopping ---
print("Training started on", DEVICE)
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask = [x.to(DEVICE) for x in batch]
        q_vec, pos_vec, neg_vec = model(q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask)
        loss = criterion(q_vec, pos_vec, neg_vec)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask = [x.to(DEVICE) for x in batch]
            q_vec, pos_vec, neg_vec = model(q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask)
            loss = criterion(q_vec, pos_vec, neg_vec)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {avg_train_loss:.4f} — Val Loss: {avg_val_loss:.4f}")

    # --- Early stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "two_tower_model.pt")
        print("✅ Model improved — saved to two_tower_model.pt")
    else:
        patience_counter += 1
        print(f"⚠️  No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("⏹️  Early stopping triggered.")
            break
