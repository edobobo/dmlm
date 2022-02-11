import argparse
import json
from typing import List

from tqdm import tqdm

from src.utils.wsd import read_from_raganato, expand_raganato_path


def convert_raganato_to_jsonl(
    raganato_paths: List[str], identifiers: List[str], output_path: str
) -> None:
    with open(output_path, "w") as f:

        for dataset_id, raganato_path in tqdm(
            zip(identifiers, raganato_paths), desc="Iterating over datasets", total=len(raganato_paths)
        ):

            for _, _, wsd_sentence in tqdm(
                read_from_raganato(*expand_raganato_path(raganato_path)),
                desc="Iterating over instances",
            ):

                tokens = [wi.annotated_token.text for wi in wsd_sentence]
                labels = [
                    wi.labels[0] if wi.labels is not None else None
                    for wi in wsd_sentence
                ]

                # if len(tokens) < 5 or len(labels) < 5:
                #     continue

                if "\n" in tokens:
                    print("ciao")

                if len(tokens) != len(labels):
                    continue

                if any(x is None for x in tokens):
                    continue

                json_dump = {
                    "sentence": " ".join(tokens),
                    "labels": [l if l is not None else "Nil" for l in labels],
                    "dataset_id": dataset_id,
                }

                f.write(json.dumps(json_dump))
                f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raganato-paths", required=True, nargs="+")
    parser.add_argument("--identifiers", required=True, nargs="+")
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    assert len(args.raganato_paths) == len(args.identifiers)
    convert_raganato_to_jsonl(args.raganato_paths, args.identifiers, args.output_path)


if __name__ == "__main__":
    main()
