import torch.nn as nn

from src.model.utils import PositionalEncoding, TransformerBlock


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)

        self.with_positional_encoding = PositionalEncoding(embed_size, max_length)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # NOTE: embedding creates the mapping from vocabulary to the embedding space
        # so input is of shape (batch_size, seq_length)
        # and output is of shape (batch_size, seq_length, embed_size)
        out = self.dropout(self.with_positional_encoding(self.word_embedding(x)))

        for transformer_block in self.transformer_blocks:
            out = transformer_block(out, out, out, mask)

        return out
