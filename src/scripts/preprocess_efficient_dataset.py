import argparse

import datasets

from src.dmlm_dataset import EfficientDMLMDataset


def preprocess_efficient_dataset(dataset_path: str, transformer_model: str, output_dir: str) -> None:

    dataset = EfficientDMLMDataset(
        dataset_path,
        inventories=None,
        transformer_model=transformer_model,
        defined_special_token="[None0]",
        definition_special_token="[None1]",
        mlm_probability=None,
        plain_mlm_probability=None,
    )

    dataset.dataset_store.save_to_disk(output_dir)

    datasets.load_from_disk(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--transformer-model", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess_efficient_dataset(args.dataset_path, args.transformer_model, args.output_dir)


if __name__ == "__main__":
    main()
