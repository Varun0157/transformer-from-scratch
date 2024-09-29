import torch


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_path():
    return "transformer_en_fr.pth"


def get_max_length():
    return 750


def get_special_tokens():
    return {
        "PAD": "<pad>",
        "SOS": "<sos>",
        "EOS": "<eos>",
        "UNK": "<unk>",
    }
