import torchtext

torchtext.disable_torchtext_deprecation_warning()

import time

import argparse
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from src.config import (
    get_hyper_details,
    get_max_length,
    get_model_path,
    get_device,
    get_special_tokens,
)
from src.transformer import Transformer


# paths
DATA_HOME = "./data/ted-talks-corpus/clean"
TRAI_EN, TRAI_FR = DATA_HOME + "/train.en", DATA_HOME + "/train.fr"
EVAL_EN, EVAL_FR = DATA_HOME + "/dev.en", DATA_HOME + "/dev.fr"
MODEL_SAVE_PATH = get_model_path()

# hypterparams
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 5e-4


def print_hyperparams(
    NUM_LAYERS: int, NUM_HEADS: int, EMB_DIM: int, DROPOUT: float
) -> None:
    print("Hyperparameters:")
    print(f"- learning rate:  {LEARNING_RATE}")
    print(f"- num layers:     {NUM_LAYERS}")
    print(f"- num heads:      {NUM_HEADS}")
    print(f"- embedding dim:  {EMB_DIM}")
    print(f"- dropout:        {DROPOUT}")
    print()


def print_training_details(
    device, batch_size, criterion, optimizer, en_vocab, fr_vocab
):
    print("Details:")
    print(f"- device:        {device}")
    print(f"- batch size:    {batch_size}")
    print(f"- criterion:     {type(criterion)}")
    print(f"- optimizer:     {type(optimizer)}")
    print(f"- eng vocab len: {len(en_vocab)}")
    print(f"- fr vocab len:  {len(fr_vocab)}")
    print()


DEVICE = get_device()

# tokenizers
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")


class TranslationDataset(Dataset):
    def __init__(self, en_path, fr_path, en_vocab, fr_vocab):
        self.en_data = open(en_path, "r", encoding="utf-8").readlines()
        self.fr_data = open(fr_path, "r", encoding="utf-8").readlines()
        assert len(self.en_data) == len(
            self.fr_data
        ), "[TranslationDataset::__init__] data mismatch"

        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab

        self.specials = get_special_tokens()

    def __len__(self):
        return len(self.en_data)

    def _create_sentence_rep(self, line, tokenizer, vocab) -> torch.Tensor:
        sos_ind = vocab[self.specials["SOS"]]
        eos_ind = vocab[self.specials["EOS"]]

        return torch.tensor(
            data=[sos_ind]
            + [vocab[token] for token in tokenizer(line.strip())]
            + [eos_ind]
        )

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._create_sentence_rep(self.en_data[idx], en_tokenizer, self.en_vocab),
            self._create_sentence_rep(self.fr_data[idx], fr_tokenizer, self.fr_vocab),
        )

    def collate_fn(self, batch):
        en_batch, fr_batch = zip(*batch)

        PAD = self.specials["PAD"]
        en_batch = nn.utils.rnn.pad_sequence(
            en_batch, padding_value=self.en_vocab[PAD], batch_first=True
        )
        fr_batch = nn.utils.rnn.pad_sequence(
            fr_batch, padding_value=self.fr_vocab[PAD], batch_first=True
        )

        return en_batch, fr_batch


def build_vocab(filepath, tokenizer):
    # creating a generator because build_vocab_from_iterator expects an iterator
    # https://pytorch.org/text/stable/vocab.html#build-vocab-from-iterator
    def yield_tokens(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield tokenizer(line.strip())

    token_dict = get_special_tokens()
    vocab = build_vocab_from_iterator(
        yield_tokens(filepath), specials=[val for val in token_dict.values()]
    )
    vocab.set_default_index(vocab[token_dict["UNK"]])

    return vocab


def train(model: Transformer, dataloader: DataLoader, optimizer, criterion):
    num_items = len(dataloader.dataset)  # type: ignore

    model.train()
    total_loss = 0

    for _, (en_batch, fr_batch) in enumerate(dataloader):
        en_batch, fr_batch = en_batch.to(DEVICE), fr_batch.to(DEVICE)

        optimizer.zero_grad()
        output = model(en_batch, fr_batch[:, :-1])  # exclude last token for input

        output = output.reshape(-1, output.shape[2])
        fr_batch = fr_batch[:, 1:].reshape(-1)  # exclude first token for target

        loss = criterion(output, fr_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_items
    return avg_loss


def eval(model: Transformer, dataloader: DataLoader, criterion):
    num_items = len(dataloader.dataset)  # type: ignore
    model.eval()
    total_loss = 0

    for _, (en_batch, fr_batch) in enumerate(dataloader):
        en_batch, fr_batch = en_batch.to(DEVICE), fr_batch.to(DEVICE)

        output = model(en_batch, fr_batch[:, :-1])  # exclude last token for input

        output = output.reshape(-1, output.shape[2])
        fr_batch = fr_batch[:, 1:].reshape(-1)  # exclude first token for target

        loss = criterion(output, fr_batch)
        total_loss += loss.item()

    avg_loss = total_loss / num_items
    return avg_loss


def save_model(model, en_vocab, fr_vocab, DROPOUT, EMB_DIM, NUM_LAYERS, NUM_HEADS):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "en_vocab": en_vocab,
            "fr_vocab": fr_vocab,
            "hyperparams": {
                "dropout": DROPOUT,
                "emb_dim": EMB_DIM,
                "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
            },
        },
        MODEL_SAVE_PATH
        + "-"
        + get_hyper_details(DROPOUT, EMB_DIM, NUM_LAYERS, NUM_HEADS)
        + ".pth",
    )


def get_train_loader(en_vocab, fr_vocab):
    train_dataset = TranslationDataset(TRAI_EN, TRAI_FR, en_vocab, fr_vocab)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    return train_loader


def get_eval_loader(en_vocab, fr_vocab):
    eval_dataset = TranslationDataset(EVAL_EN, EVAL_FR, en_vocab, fr_vocab)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=eval_dataset.collate_fn,
    )

    return eval_loader


def train_and_save(
    DROPOUT: float,
    EMB_DIM: int,
    NUM_LAYERS: int,
    NUM_HEADS: int,
):
    # Build vocabularies
    en_vocab = build_vocab(TRAI_EN, en_tokenizer)
    fr_vocab = build_vocab(TRAI_FR, fr_tokenizer)

    print_hyperparams(NUM_LAYERS, NUM_HEADS, EMB_DIM, DROPOUT)

    # create dataloaders
    train_loader = get_train_loader(en_vocab, fr_vocab)
    eval_loader = get_eval_loader(en_vocab, fr_vocab)

    # initialise model
    model = Transformer(
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(fr_vocab),
        src_pad_idx=en_vocab[get_special_tokens()["PAD"]],
        trg_pad_idx=fr_vocab[get_special_tokens()["PAD"]],
        device=DEVICE,
        max_length=get_max_length(),
        dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        heads=NUM_HEADS,
        embed_size=EMB_DIM,
    ).to(DEVICE)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss(
        ignore_index=fr_vocab[get_special_tokens()["PAD"]], reduction="sum"
    )
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print_training_details(DEVICE, BATCH_SIZE, criterion, optimizer, en_vocab, fr_vocab)

    # training loop
    best_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion)
        eval_loss = eval(model, eval_loader, criterion)
        epoch_time = time.time() - start_time

        print(
            f"epoch {epoch} -> train loss: {train_loss:.4f}, eval loss: {eval_loss:.4f}\ttime: {epoch_time:.2f}s"
        )

        if eval_loss >= best_loss:
            continue

        best_loss = eval_loss
        save_model(model, en_vocab, fr_vocab, DROPOUT, EMB_DIM, NUM_LAYERS, NUM_HEADS)
        print(f"\tmodel saved")

    print("\ntraining complete.")


def main():
    parser = argparse.ArgumentParser(description="Transformer training script")
    parser.add_argument("--dropout", type=float, required=True, help="dropout rate")
    parser.add_argument(
        "--embed_dim", type=int, required=True, help="embedding dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, required=True, help="number of layers"
    )
    parser.add_argument(
        "--num_heads", type=int, required=True, help="number of attention heads"
    )

    # Parse the arguments
    args = parser.parse_args()

    assert args.dropout >= 0 and args.dropout <= 1, "dropout must be between 0 and 1"
    assert args.embed_dim > 0, "embedding dimension must be greater than 0"
    assert args.num_layers > 0, "number of layers must be greater than 0"
    assert args.num_heads > 0, "number of heads must be greater than 0"

    # Call the train_and_save function with the parsed arguments
    train_and_save(
        DROPOUT=args.dropout,
        EMB_DIM=args.embed_dim,
        NUM_LAYERS=args.num_layers,
        NUM_HEADS=args.num_heads,
    )


if __name__ == "__main__":
    main()
