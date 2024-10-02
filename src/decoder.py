import torch.nn as nn

from src.utils import PositionalEncoding, SelfAttention, TransformerBlock


class DecoderBlock(nn.Module):
    def __init__(
        self, embed_size: int, heads: int, forward_expansion: int, dropout: float
    ):
        super(DecoderBlock, self).__init__()
        # masked multi head attention
        self.self_attention = SelfAttention(embed_size, heads)
        # NOTE: why layer norm and not batch norm? check the TransformerBlock class in utils.py
        self.add_and_norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.self_attention(x, x, x, trg_mask)
        # NOTE: here, we mask out the future tokens using trg_mask, for the output embeddings to the decoder
        masked_out = self.dropout(
            self.add_and_norm(attention + x)
        )  # NOTE: residual connection

        out = self.transformer_block(value, key, masked_out, src_mask)
        # NOTE:
        # - masked_out is the query vals of the decoder, keys and vals from the encoder
        # - for the rest, we use the src_mask to mask out the padding tokens

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.with_positional_embedding = PositionalEncoding(embed_size, max_length)

        self.dropout = nn.Dropout(dropout)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(embed_size, trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, enc_out, src_mask, trg_mask):
        x = self.dropout(self.with_positional_embedding(self.word_embedding(x)))

        for decoder_block in self.decoder_blocks:
            # NOTE: queries from decoder, keys and values from encoder
            x = decoder_block(x, enc_out, enc_out, src_mask, trg_mask)
            # NOTE: we do not feed back the output to the next layer during training
            #   we just use the mask to ensure we consider the required tokens per layer
            #   of the stack. We do have to feed it back during inference, however.

        out = self.linear(x)
        return self.softmax(out)
