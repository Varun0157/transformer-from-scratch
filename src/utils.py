from math import log

import torch
import torch.nn as nn

NEG_INF = float("-1e20")  # prevents numerical overflow, rather than using -inf


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        assert (
            self.embed_size % self.heads
        ) == 0, "[SelfAttention::init] embed size needs to be divisible by heads"

        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(
        self,
        values: torch.Tensor,
        keys: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        # einsum is easier than having to perform the calculations and determine the
        #   batch multiplication (torch.bmm) because we have n batches
        # query shape:  (N, query_len, heads, head_dim)
        # keys shape:   (N, key_len, heads, head_dim)
        # goal:         (N, heads, query_len, key_len)
        energy = torch.einsum(
            "nqhd,nkhd->nhqk", [query, keys]
        )  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, NEG_INF)

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # dim = 3 because we're normalising across the key_len
        # todo: why normalise across the key_len

        # attention shape:  (N, heads, query_len, key_len)
        # values shape:     (N, value_len, heads, head_dim)
        # einsum goal:      (N, query_len, heads, head_dim)
        assert key_len == value_len, "[SelfAttention::forward] key_len != value_len"

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self, embed_size: int, heads: int, dropout: float, forward_expansion: int
    ):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size=embed_size, heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # LayerNorm: normalises the output of the attention block
        # todo: why LayerNorm and not BatchNorm

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        value: torch.Tensor,
        key: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        # todo: refactor

        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))  # NOTE: residual connection

        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))  # NOTE: residual connection

        return out


# http://nlp.seas.harvard.edu/annotated-transformer/#positional-encoding
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
