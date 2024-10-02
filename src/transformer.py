import torch
import torch.nn as nn

from src.decoder import Decoder
from src.encoder import Encoder


# major source: https://www.youtube.com/watch?v=U0s0f995w14
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=240,
        num_layers=3,
        forward_expansion=4,
        heads=3,
        dropout=0.2,
        device: torch.device = torch.device("cpu"),
        max_length=750,  # todo: check, and generalize
    ):
        super(Transformer, self).__init__()
        self.device = device

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # shape: (N, 1, 1, src_len)
        # used in the transformer_block to mask out the padding tokens

        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape  # (batch_size, trg_len)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
