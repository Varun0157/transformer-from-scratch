import os
import time

import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from src.config import get_max_length, get_model_path, get_device, get_special_tokens
from src.transformer import Transformer


# Paths
DATA_HOME = "./data/ted-talks-corpus"
TRAI_EN, TRAI_FR = DATA_HOME + "/train.en", DATA_HOME + "/train.fr"
EVAL_EN, EVAL_FR = DATA_HOME + "/dev.en", DATA_HOME + "/dev.fr"
MODEL_SAVE_PATH = get_model_path()

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_LAYERS = 3
NUM_HEADS = 2
EMB_DIM = 240
DEVICE = get_device()


def print_hyperparams() -> None:
    print("Hyperparameters:")
    print(f"- device:         {DEVICE}")
    print(f"- learning rate:  {LEARNING_RATE}")
    print(f"- num layers:     {NUM_LAYERS}")
    print(f"- num heads:      {NUM_HEADS}")
    print(f"- embedding dim:  {EMB_DIM}")
    print()


# Tokenizers
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")


class TranslationDataset(Dataset):
    def __init__(self, en_path, fr_path, en_vocab, fr_vocab):
        self.en_data = open(en_path, "r", encoding="utf-8").readlines()
        self.fr_data = open(fr_path, "r", encoding="utf-8").readlines()
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab

        self.specials = get_special_tokens()

    def __len__(self):
        return len(self.en_data)

    def wrap_sentence(self, sentence: str) -> str:
        return " ".join([self.specials["SOS"], sentence.strip(), self.specials["EOS"]])

    def __getitem__(self, idx):
        en_text = self.wrap_sentence(self.en_data[idx])
        fr_text = self.wrap_sentence(self.fr_data[idx])

        en_tensor = torch.tensor(
            [self.en_vocab[token] for token in en_tokenizer(en_text)]
        )
        fr_tensor = torch.tensor(
            [self.fr_vocab[token] for token in fr_tokenizer(fr_text)]
        )

        return en_tensor, fr_tensor

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
    model.train()
    total_loss = 0

    for _, (en_batch, fr_batch) in enumerate(dataloader):
        en_batch, fr_batch = en_batch.to(DEVICE), fr_batch.to(DEVICE)

        optimizer.zero_grad()
        output = model(en_batch, fr_batch[:, :-1])

        output = output.reshape(-1, output.shape[2])
        fr_batch = fr_batch[:, 1:].reshape(-1)

        loss = criterion(output, fr_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def eval(model: Transformer, dataloader: DataLoader, criterion):
    model.eval()
    total_loss = 0

    for _, (en_batch, fr_batch) in enumerate(dataloader):
        en_batch, fr_batch = en_batch.to(DEVICE), fr_batch.to(DEVICE)

        output = model(en_batch, fr_batch[:, :-1])

        output = output.reshape(-1, output.shape[2])
        fr_batch = fr_batch[:, 1:].reshape(-1)

        loss = criterion(output, fr_batch)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    # Build vocabularies
    en_vocab = build_vocab(TRAI_EN, en_tokenizer)
    print("english vocab size : ", len(en_vocab))
    fr_vocab = build_vocab(TRAI_FR, fr_tokenizer)
    print("french vocab size : ", len(fr_vocab))

    def save_model(model, en_vocab=en_vocab, fr_vocab=fr_vocab):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "en_vocab": en_vocab,
                "fr_vocab": fr_vocab,
            },
            MODEL_SAVE_PATH,
        )

    os.system("cls || clear")
    print_hyperparams()

    # Create dataset and dataloader
    dataset = TranslationDataset(TRAI_EN, TRAI_FR, en_vocab, fr_vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Initialize model
    PAD = get_special_tokens()["PAD"]
    model = Transformer(
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(fr_vocab),
        src_pad_idx=en_vocab[PAD],
        trg_pad_idx=fr_vocab[PAD],
        device=DEVICE,
        max_length=get_max_length(),
    ).to(DEVICE)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=fr_vocab[PAD])
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print("Details:")
    print(f"- batch size:   {BATCH_SIZE}")
    print(f"- criterion:    {criterion}")
    print(f"- optimizer:    {optimizer}")
    print()

    best_loss = float("inf")

    # training loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        train_loss = train(model, dataloader, optimizer, criterion)
        eval_loss = eval(model, dataloader, criterion)
        epoch_time = time.time() - start_time

        print(
            f"epoch {epoch} -> train loss: {train_loss:.4f}, eval loss: {eval_loss:.4f}\ttime: {epoch_time:.2f}s"
        )

        if eval_loss >= best_loss:
            continue

        best_loss = eval_loss
        save_model(model)
        print(f"\tmodel saved with loss: {best_loss:.4f}")

    print("\ntraining complete.")


if __name__ == "__main__":
    main()
