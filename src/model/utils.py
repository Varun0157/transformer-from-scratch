from math import log

import torch
import torch.nn as nn

NEG_INF = float("-1e30")  # prevents numerical overflow, rather than using -inf


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        assert (
            self.embed_size % self.heads
        ) == 0, "[SelfAttention::__init__] embed size needs to be divisible by heads"
        # NOTE: this is because we're splitting the embedding into self.heads pieces

        self.head_dim = embed_size // heads

        self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # NOTE: why bias=False? these are meant to be weight matrices in the paper, making the
        #   input undergo a pure linear transform.
        # there is bias in later layers anyway.

        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        # NOTE: heads * self.head_dim = embed_size, but I'm writing it explicitly for clarity

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # shape of values, keys, query: (N, seq_len, embed_size)
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split the embedding into self.heads pieces
        values = self.W_v(values.reshape(N, value_len, self.heads, self.head_dim))
        keys = self.W_k(keys.reshape(N, key_len, self.heads, self.head_dim))
        query = self.W_q(query.reshape(N, query_len, self.heads, self.head_dim))

        # einsum is easier than having to perform the calculations and determine the
        #   batch multiplication (torch.bmm) because we have n items in a batch
        # query shape:  (N, query_len, heads, head_dim)
        # keys shape:   (N, key_len,   heads, head_dim)
        # goal:         (N, heads, query_len, key_len)
        QK_T = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        # NOTE: for each batch term, for each head, we want the Q.K^T matrix.
        #  that should make the dimensions make sense

        if mask is not None:
            QK_T = QK_T.masked_fill(mask == 0, NEG_INF)
            # (q_len, k_len). because of our lower triangular mask, the way it works
            #   is that the first row can only attend to itself, the second row can
            #   attend to the first and itself, and so on.

        QK_T_scaled = torch.softmax(QK_T / (self.embed_size ** (1 / 2)), dim=3)
        # NOTE: dim = 3 is to normalise the scores of each query across all keys
        #   (i.e. the softmax is applied to the last dimension)

        # QK_T_scaled shape:    (N, heads, query_len, key_len)
        # values shape:         (N, value_len, heads, head_dim)
        # einsum goal:          (N, query_len, heads, head_dim)
        assert key_len == value_len, "[SelfAttention::forward] key_len != value_len"

        self_attentions = torch.einsum(
            "nhql,nlhd->nqhd", [QK_T_scaled, values]
        ).reshape(N, query_len, self.heads * self.head_dim)
        # NOTE: for each batch term, for each query, we want the attention output
        # QK_T essentially gives (q_len, k_len), V is (v_len, emb_dim)
        #   thus, the expected output is the embeddings for each query

        mha = self.fc_out(self_attentions)
        return mha


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_size: int, heads: int, dropout: float, forward_expansion: int
    ):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        # LayerNorm: normalises the output of the attention block
        # NOTE: https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        attention = self.attention(value, key, query, mask)
        out = self.dropout(self.norm1(attention + query))  # NOTE: residual connection

        forward = self.feed_forward(out)
        out = self.dropout(self.norm2(forward + out))  # NOTE: residual connection

        return out


# http://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding
# NOTE: removed the dropout from the above link, did not seem to be mentioned in many places
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len: int):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
