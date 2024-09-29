import torch
from transformer import Transformer
from torchtext.data.utils import get_tokenizer
from nltk.translate.bleu_score import sentence_bleu
import nltk

# Download NLTK data for BLEU score calculation
nltk.download("punkt")

# Paths
TEST_EN = "data/test-en.txt"
TEST_FR = "data/test-fr.txt"
MODEL_LOAD_PATH = "transformer_en_fr.pth"
OUTPUT_FILE = "translations_with_bleu.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    checkpoint = torch.load(MODEL_LOAD_PATH)
    en_vocab = checkpoint["en_vocab"]
    fr_vocab = checkpoint["fr_vocab"]

    model = Transformer(
        src_vocab_size=len(en_vocab),
        trg_vocab_size=len(fr_vocab),
        src_pad_idx=en_vocab["<pad>"],
        trg_pad_idx=fr_vocab["<pad>"],
        device=DEVICE,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, en_vocab, fr_vocab


def translate_sentence(
    model, sentence, en_vocab, fr_vocab, en_tokenizer, max_length=750
):
    model.eval()

    tokens = ["<sos>"] + en_tokenizer(sentence) + ["<eos>"]
    src_indexes = [en_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(DEVICE)

    src_mask = model.make_src_mask(src_tensor)

    enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [fr_vocab["<sos>"]]

    for _ in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(DEVICE)
        trg_mask = model.make_trg_mask(trg_tensor)

        output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)
        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == fr_vocab["<eos>"]:
            break

    trg_tokens = [fr_vocab.get_itos()[i] for i in trg_indexes]
    return " ".join(trg_tokens[1:-1])  # Remove <sos> and <eos>


def calculate_bleu(reference, hypothesis):
    return sentence_bleu([reference.split()], hypothesis.split())


def main():
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    model, en_vocab, fr_vocab = load_model()

    with open(TEST_EN, "r", encoding="utf-8") as en_file, open(
        TEST_FR, "r", encoding="utf-8"
    ) as fr_file, open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:

        for en_sentence, fr_sentence in zip(en_file, fr_file):
            en_sentence = en_sentence.strip()
            fr_sentence = fr_sentence.strip()

            translation = translate_sentence(
                model, en_sentence, en_vocab, fr_vocab, en_tokenizer
            )
            bleu_score = calculate_bleu(fr_sentence, translation)

            out_file.write(f"Source: {en_sentence}\n")
            out_file.write(f"Target: {fr_sentence}\n")
            out_file.write(f"Translation: {translation}\n")
            out_file.write(f"BLEU Score: {bleu_score:.4f}\n\n")

            print(f"Processed: {en_sentence}")
            print(f"BLEU Score: {bleu_score:.4f}")
            print()


if __name__ == "__main__":
    main()
