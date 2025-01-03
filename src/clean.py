import os
import logging

import re
from string import punctuation as PUNCTUATION


def _clean(sentence: str) -> str:
    sentence = sentence.lower().strip()
    # remove all text that lies between square brackets using regex - because there seem to be random citations
    sentence = re.sub(r"\[.*?\]", "", sentence)

    for ch in "'":
        sentence = sentence.replace(ch, "")

    for ch in '"”—“’‘' + PUNCTUATION:  # removed - for instances like well-known
        sentence = sentence.replace(ch, " ")

    # remove all words that contain any numbers
    sentence = " ".join(
        [word for word in sentence.split() if not any(char.isdigit() for char in word)]
    )

    return sentence


def clean_file(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as in_file, open(
        out_path, "w", encoding="utf-8"
    ) as out_file:
        for line in in_file:
            out_file.write(_clean(line) + "\n")
    logging.info(f"{in_path} cleaned and saved to {out_path}")


def main(DATA_DIR="./data/ted-talks-corpus"):
    CLEAN_DIR = DATA_DIR + os.path.sep + "clean"
    if not os.path.exists(CLEAN_DIR):
        os.makedirs(CLEAN_DIR)

    items = os.listdir(DATA_DIR)

    for item in items:
        item_path = f"{DATA_DIR + os.path.sep + item}"
        if not os.path.isfile(item_path):
            continue
        clean_file(item_path, f"{CLEAN_DIR + os.path.sep + item}")
    logging.info(f"\n->files in {DATA_DIR} cleaned and saved")


if __name__ == "__main__":
    main()
