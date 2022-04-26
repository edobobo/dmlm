import argparse
import json
import os
from typing import List

from tokenizers.processors import TemplateProcessing
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

import tempfile


def inline_preprocesses_files(
    preprocessed_files: List[str], tmp_dir_path: str
) -> List[str]:
    out_files_paths = [
        f"{tmp_dir_path}/file_{i}.txt" for i in range(len(preprocessed_files))
    ]

    for in_file_path, out_file_path in tqdm(
        zip(preprocessed_files, out_files_paths), desc=f"Processing input files"
    ):
        with open(in_file_path) as fin, open(out_file_path, "w") as fout:
            for line in tqdm(fin, desc="Reading lines..."):
                json_obj = json.loads(line.strip())
                fout.write(json_obj["sentence"])
                fout.write("\n")

    print("'head -2' on each tmp file")
    for ofp in out_files_paths:
        os.system(f"head -2 {ofp}")

    print("total number of lines per file")
    for ofp in out_files_paths:
        os.system(f"{ofp}: wc -l {ofp}")

    return out_files_paths


def train_tokenizer(input_files: List[str], output_path: str):
    """
    input_files: List[str] -> preprocessed jsonl files with the "sentence" field in each line
    """

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        inline_files = inline_preprocesses_files(input_files, tmp_dir_name)

        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            special_tokens=[
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[PAD]",
                "[MASK]",
                "[DEF]",
                "[DEFINITION]",
            ],
            vocab_size=28996,  # bert one
        )

        tokenizer.pre_tokenizer = Whitespace()

        tokenizer.train(inline_files, trainer)

        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

        tokenizer.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    train_tokenizer(args.input_files, args.output_path)


if __name__ == "__main__":
    main()
