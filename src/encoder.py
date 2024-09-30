import torch.nn as nn

from src.utils import PositionalEncoding, TransformerBlock


# todo: replace the vocab embedding with torchtext
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

        # todo: replace with sinusoidal positional encoding
        self.with_positional_encoding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.dropout(self.with_positional_encoding(self.word_embedding(x)))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
