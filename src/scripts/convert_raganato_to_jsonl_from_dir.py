import os
import argparse
from typing import List

from itertools import groupby

import numpy as np

from src.scripts.convert_raganato_to_jsonl import convert_raganato_to_jsonl


def pick_random_files(input_dir: str, n_files: int) -> List[str]:
    files_in_dir = os.listdir(input_dir)
    files_in_dir = [
        ".".join(_f.split(".")[:-2]) for _f in files_in_dir if "data.xml" in _f
    ]
    chosen_files = np.random.choice(files_in_dir, n_files)
    return [f"{input_dir}/{cf}" for cf in chosen_files]


def pick_files_and_convert_raganato_to_jsonl(
    input_dir: str, chunks_per_inventory: int, output_path: str
):

    selected_files = []
    corresponding_inventories = []
    for inventory in ["oxford", "wiktionary", "wordnet"]:
        random_files = pick_random_files(
            f"{input_dir}/{inventory}", chunks_per_inventory
        )
        selected_files += random_files
        corresponding_inventories += [inventory] * len(random_files)

    for key, group in groupby(
        zip(selected_files, corresponding_inventories), lambda x: x[1]
    ):
        files_in_group = [x[0] for x in group]
        print(
            f"Inventory: {key} | ({len(files_in_group)}) Files: {', '.join(files_in_group)}"
        )

    convert_raganato_to_jsonl(selected_files, corresponding_inventories, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--chunks-per-inventory", required=True, type=int)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pick_files_and_convert_raganato_to_jsonl(
        args.input_dir, args.chunks_per_inventory, args.output_path
    )


if __name__ == "__main__":
    main()
