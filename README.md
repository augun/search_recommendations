# 🔍 Semantic Search with Two-Tower Architecture (MS MARCO)

This project implements a document retrieval system using a dual-encoder (Two-Tower) model trained on the MS MARCO dataset. It uses pretrained GloVe embeddings, triplet loss, and cosine similarity to match queries with relevant documents.

---

## 🚀 Features
- Dual-encoder architecture with GRU encoders
- Pretrained GloVe embeddings
- Hard negative mining across samples
- Triplet loss optimization
- Early stopping with validation monitoring
- Cached document embeddings for fast query search

---

## 🗂️ File Overview

| File                          | Description |
|-------------------------------|-------------|
| `preprocess_to_tensor.py`    | Preprocesses MS MARCO into train/val triplet tensors with smarter negatives |
| `train.py`                   | Trains TwoTowerModel with early stopping, saves best model |
| `search_in_cached.py`        | Loads model and cached docs, performs interactive top-k semantic search |
| `load_glove_embeddings.py`   | Converts GloVe `.txt` file to PyTorch tensor embedding matrix |
| `two_tower.py`               | Model definition: GRU-based encoder towers with frozen embeddings |
| `glove_embeddings.pt`        | Saved tensor of GloVe vectors used during training/inference |
| `train_tensor.pt` / `val_tensor.pt` | Tensors for training and validation sets |
| `two_tower_model.pt`         | Final trained model checkpoint |
| `cached_doc_vectors.pt`      | Precomputed document embeddings for fast vector search |

---

## ⚙️ Setup
```bash
pip install -r requirements.txt
```
Also download and unzip GloVe:
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
python load_glove_embeddings.py  # produces glove_embeddings.pt
```

---

## 🏋️ Training
```bash
python preprocess_to_tensor.py   # Generates train_tensor.pt and val_tensor.pt
python train.py                  # Trains the model with early stopping
```

---

## 🔍 Searching
```bash
python search_in_cached.py       # Interactive query terminal
```

Example:
```
🔍 Enter your query: what is rba
Top 5 results:
1. [Score: 0.8721] Results-Based Accountability (RBA) is a disciplined way of thinking...
```

---

## 📌 Notes
- You can scale to more documents by expanding `sample_passages` in `search_in_cached.py`
- For faster retrieval at scale, consider integrating FAISS
- You can build a frontend using Streamlit or Gradio

---

## ✨ Future Ideas
- Add FAISS for nearest-neighbor search
- Plug in a RAG pipeline with LLM (e.g., Mistral, OpenAI)
- Visualize embeddings using TensorBoard projector or UMAP

---

**Built by Aygun with guidance from ChatGPT 💙**

