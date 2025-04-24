import torch
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm


def load_glove_embeddings(glove_path, tokenizer, embedding_dim=300):
    """
    Loads GloVe vectors and returns a tensor for use in nn.Embedding.from_pretrained
    """
    print("Loading GloVe vectors...")
    glove_vectors = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if len(vector) == embedding_dim:
                glove_vectors[word] = vector

    vocab_size = tokenizer.vocab_size
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim)).astype(np.float32)

    print("Building embedding matrix...")
    for token, idx in tokenizer.get_vocab().items():
        if token in glove_vectors:
            embedding_matrix[idx] = glove_vectors[token]
        elif token.startswith("##") and token[2:] in glove_vectors:
            embedding_matrix[idx] = glove_vectors[token[2:]]
        elif token.lower() in glove_vectors:
            embedding_matrix[idx] = glove_vectors[token.lower()]

    return torch.tensor(embedding_matrix)


# --- Example usage ---
if __name__ == "__main__":
    glove_file = "glove.6B.300d.txt"  # Path to downloaded GloVe
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedding_tensor = load_glove_embeddings(glove_file, tokenizer)
    torch.save(embedding_tensor, "glove_embeddings.pt")
    print("✅ Saved embedding tensor to glove_embeddings.pt")
