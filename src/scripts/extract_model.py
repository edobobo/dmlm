import os
from typing import Optional

import argparse

from src.pl_modules import TransformerDMLM


def extract_transformer(
    model_path: str, output_dir: str, member_variable: Optional[str] = None
) -> None:
    pl_module = TransformerDMLM.load_from_checkpoint(model_path)

    if member_variable is None:
        member_variable = "model"

    transformer_model = getattr(pl_module, member_variable, None)

    if transformer_model is None:
        print(
            f"The member variable '{member_variable}' does not exist in the provided model_path {model_path}"
        )

    transformer_model.save_pretrained(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--member-variable")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    extract_transformer(args.model_path, args.output_dir, args.member_variable)


if __name__ == "__main__":
    main()
