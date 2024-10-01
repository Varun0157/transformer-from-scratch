import torchtext

torchtext.disable_torchtext_deprecation_warning()

import torch
from torchtext.data.utils import get_tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.config import get_device, get_max_length, get_model_path, get_special_tokens
from src.transformer import Transformer

nltk.download("punkt")

# config details
DATA_HOME = "./data/ted-talks-corpus/clean"
TEST_EN = DATA_HOME + "/test.en"
TEST_FR = DATA_HOME + "/test.fr"
MODEL_LOAD_PATH = get_model_path()
OUTPUT_FILE = "translations_with_bleu.txt"

DEVICE = get_device()


def load_model():
    checkpoint = torch.load(MODEL_LOAD_PATH)
    en_vocab = checkpoint["en_vocab"]
    fr_vocab = checkpoint["fr_vocab"]
    hyperparams = checkpoint["hyperparams"]

    special_tokens = get_special_tokens()
    model = Transformer(
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(fr_vocab),
        src_pad_idx=en_vocab[special_tokens["PAD"]],
        trg_pad_idx=fr_vocab[special_tokens["PAD"]],
        embed_size=hyperparams["emb_dim"],
        num_layers=hyperparams["num_layers"],
        heads=hyperparams["num_heads"],
        dropout=hyperparams["dropout"],
        max_length=get_max_length(),
        device=DEVICE,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, en_vocab, fr_vocab


def translate_sentence(
    model, sentence, en_vocab, fr_vocab, en_tokenizer, max_length=get_max_length()
) -> list[str]:
    model.eval()

    start_of_sent, end_of_sent = (
        get_special_tokens()["SOS"],
        get_special_tokens()["EOS"],
    )
    tokens = [start_of_sent] + en_tokenizer(sentence) + [end_of_sent]
    src_indexes = [en_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)

    src_mask = model.make_src_mask(src_tensor)

    enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [fr_vocab[start_of_sent]]

    for _ in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)
        trg_mask = model.make_trg_mask(trg_tensor)

        output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == fr_vocab[end_of_sent]:
            break

    trg_tokens = [fr_vocab.get_itos()[i] for i in trg_indexes]
    return trg_tokens


def calculate_bleu(reference, hypothesis) -> float:
    smo_func = SmoothingFunction().method7
    scores = sentence_bleu([reference], hypothesis, smoothing_function=smo_func)
    assert type(scores) == float
    return scores


# todo: can make this faster by reading in batches probably
def main():
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    fr_tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")

    model, en_vocab, fr_vocab = load_model()

    start_of_sent, end_of_sent = (
        get_special_tokens()["SOS"],
        get_special_tokens()["EOS"],
    )

    with open(TEST_EN, "r", encoding="utf-8") as en_file, open(
        TEST_FR, "r", encoding="utf-8"
    ) as fr_file, open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        num_sent = 0
        total_bleu = 0

        def write_and_print(line):
            out_file.write(line)
            print(line)

        for index, (en_sentence, fr_tokens) in enumerate(zip(en_file, fr_file)):
            fr_tokens = (
                [start_of_sent] + fr_tokenizer(fr_tokens.strip()) + [end_of_sent]
            )

            tr_tokens = translate_sentence(
                model, en_sentence.strip(), en_vocab, fr_vocab, en_tokenizer
            )
            bleu_score = calculate_bleu(fr_tokens, tr_tokens)
            total_bleu += bleu_score
            num_sent += 1

            out_file.write(f"Source:        {en_sentence}\n")
            out_file.write(f"Target:        {fr_tokens}\n")
            out_file.write(f"Translation:   {tr_tokens}\n")
            out_file.write(f"BLEU Score:    {bleu_score:.4f}\n\n")

            write_and_print(f"{index} -> <{en_sentence}> <{bleu_score:.4f}>")

        assert num_sent != 0
        avg_bleu = total_bleu / num_sent
        write_and_print(f"\n\n--> average score: {avg_bleu:.4f}")


if __name__ == "__main__":
    main()
