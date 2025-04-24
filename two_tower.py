import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_dim):
        super().__init__()
        self.embedding = embedding
        self.rnn = nn.GRU(embedding.embedding_dim, hidden_dim, batch_first=True)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        masked = embedded * attention_mask.unsqueeze(-1)
        _, hidden = self.rnn(masked)
        return hidden.squeeze(0)

class TwoTowerModel(nn.Module):
    def __init__(self, embedding, hidden_dim):
        super().__init__()
        self.query_encoder = Encoder(embedding, hidden_dim)
        self.doc_encoder = Encoder(embedding, hidden_dim)

    def encode_query(self, ids, mask):
        return self.query_encoder(ids, mask)

    def encode_document(self, ids, mask):
        return self.doc_encoder(ids, mask)

    def forward(self, q_ids, q_mask, pos_ids, pos_mask, neg_ids, neg_mask):
        q_vec = self.encode_query(q_ids, q_mask)
        pos_vec = self.encode_document(pos_ids, pos_mask)
        neg_vec = self.encode_document(neg_ids, neg_mask)
        return q_vec, pos_vec, neg_vec
