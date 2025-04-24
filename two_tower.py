import torch
import torch.nn as nn

class TowerEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, embedding_weights=None, freeze_embeddings=False):
        super(TowerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if embedding_weights is not None:
            self.embedding.weight = nn.Parameter(embedding_weights)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False  # set True if you want bidirectional encoding
        )

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        outputs, _ = self.rnn(embedded)       # [batch, seq_len, hidden_dim]

        # Apply mask to zero out padding outputs
        mask = attention_mask.unsqueeze(-1).type_as(outputs)  # [batch, seq_len, 1]
        masked_outputs = outputs * mask

        # Average over the sequence length (only non-padding tokens)
        summed = masked_outputs.sum(dim=1)  # [batch, hidden_dim]
        count = mask.sum(dim=1).clamp(min=1e-9)  # avoid division by zero
        mean_pooled = summed / count

        return mean_pooled  # shape: [batch, hidden_dim]


class TwoTowerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, embedding_weights=None, freeze_embeddings=False):
        super(TwoTowerModel, self).__init__()
        self.query_encoder = TowerEncoder(vocab_size, embedding_dim, hidden_dim, embedding_weights, freeze_embeddings)
        self.doc_encoder = TowerEncoder(vocab_size, embedding_dim, hidden_dim, embedding_weights, freeze_embeddings)

    def forward(self, query_input_ids, query_attention_mask,
                pos_input_ids, pos_attention_mask,
                neg_input_ids, neg_attention_mask):

        # Encode each input
        query_vec = self.query_encoder(query_input_ids, query_attention_mask)  # (batch_size, hidden_dim)
        pos_vec = self.doc_encoder(pos_input_ids, pos_attention_mask)
        neg_vec = self.doc_encoder(neg_input_ids, neg_attention_mask)

        return query_vec, pos_vec, neg_vec

    def encode_query(self, input_ids, attention_mask):
        return self.query_encoder(input_ids, attention_mask)

    def encode_document(self, input_ids, attention_mask):
        return self.doc_encoder(input_ids, attention_mask)
