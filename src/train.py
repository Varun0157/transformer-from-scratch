import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from src.transformer import Transformer
import os

# Paths
DATA_HOME = "./data/ted-talks-corpus"
TRAI_EN, TRAI_FR = DATA_HOME + "/train.en", DATA_HOME + "/train.fr"
EVAL_EN, EVAL_FR = DATA_HOME + "/dev.en", DATA_HOME + "/dev.fr"
MODEL_SAVE_PATH = "transformer_en_fr.pth"

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizers
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")


class TranslationDataset(Dataset):
    def __init__(self, en_path, fr_path, en_vocab, fr_vocab):
        self.en_data = open(en_path, "r", encoding="utf-8").readlines()
        self.fr_data = open(fr_path, "r", encoding="utf-8").readlines()
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab

    def __len__(self):
        return len(self.en_data)

    def __getitem__(self, idx):
        en_text = self.en_data[idx].strip()
        fr_text = self.fr_data[idx].strip()

        en_tensor = torch.tensor(
            [self.en_vocab[token] for token in en_tokenizer(en_text)]
        )
        fr_tensor = torch.tensor(
            [self.fr_vocab[token] for token in fr_tokenizer(fr_text)]
        )

        return en_tensor, fr_tensor

    def collate_fn(self, batch):
        en_batch, fr_batch = zip(*batch)

        en_batch = nn.utils.rnn.pad_sequence(
            en_batch, padding_value=self.en_vocab["<pad>"], batch_first=True
        )
        fr_batch = nn.utils.rnn.pad_sequence(
            fr_batch, padding_value=self.fr_vocab["<pad>"], batch_first=True
        )

        return en_batch, fr_batch


def build_vocab(filepath, tokenizer):
    def yield_tokens(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield tokenizer(line.strip())

    vocab = build_vocab_from_iterator(
        yield_tokens(filepath), specials=["<unk>", "<pad>", "<sos>", "<eos>"]
    )
    vocab.set_default_index(vocab["<unk>"])
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
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and False else "cpu")

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

    # Create dataset and dataloader
    dataset = TranslationDataset(TRAI_EN, TRAI_FR, en_vocab, fr_vocab)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Initialize model
    model = Transformer(
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(fr_vocab),
        src_pad_idx=en_vocab["<pad>"],
        trg_pad_idx=fr_vocab["<pad>"],
        device=DEVICE,
    ).to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=fr_vocab["<pad>"])
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float("inf")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"epoch {epoch+1}/{NUM_EPOCHS}")

        train_loss = train(model, dataloader, optimizer, criterion)

        eval_loss = eval(model, dataloader, criterion)
        print(
            f"epoch {epoch} -> train loss: {train_loss:.4f}, eval loss: {eval_loss:.4f}"
        )

        if eval_loss >= best_loss:
            continue

        best_loss = eval_loss
        save_model(model)
        print(f"\tmodel saved with loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
