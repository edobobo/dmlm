from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from src.utils.wsd import WSDInstance, expand_raganato_path, read_from_raganato


class DMLMDataset(Dataset):
    def __init__(
        self,
        data2inventories: Dict[str, str],
        inventory_paths: Dict[str, str],
        transformer_model: str,
        defined_special_token: str,
        definition_special_token: str,
        mlm_probability: float,
    ):
        self.datasets: List[List[List[WSDInstance]]] = []
        self.datasets_inventory: List[str] = []
        self.inventories: Dict[str, Dict[str, str]] = dict()
        self.sense_inverse_frequencies: Dict[str, float] = dict()

        self.defined_special_token = defined_special_token
        self.definition_special_token = definition_special_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model,
            additional_special_tokens=[defined_special_token, definition_special_token],
        )

        self.mlm_probability = mlm_probability

        self.final_dataset: List[Dict[str, Any]] = []

        self._init_datasets_and_inventories(data2inventories, inventory_paths)
        self._init_sense_inverse_frequencies()
        self._init_final_dataset()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.final_dataset[index]

    def _init_datasets_and_inventories(
        self, data2inventory_paths: Dict[str, str], inventory_paths: Dict[str, str]
    ) -> None:
        def load_dataset(dst_path: str) -> List[List[WSDInstance]]:
            return [x[-1] for x in read_from_raganato(*expand_raganato_path(dst_path))]

        def load_inventory(inv_path: str) -> Dict[str, str]:
            inventory_store = dict()
            with open(inv_path) as f:
                for line in f:
                    sense_key, _, _, definition = line.strip().split("\t")
                    inventory_store[sense_key] = definition
            return inventory_store

        print("Initializing inventories...")
        for inventory_name, inventory_path in inventory_paths.items():
            self.inventories[inventory_name] = load_inventory(inventory_path)

        print("Initializing datasets...")
        for dataset_path, inventory_name in data2inventory_paths.items():
            self.datasets.append(load_dataset(dataset_path))
            self.datasets_inventory.append(inventory_name)

    def _init_sense_inverse_frequencies(self) -> None:
        print("Computing senses inverse frequencies")
        inventory2sense_count: Dict[str, Counter] = defaultdict(Counter)

        def update_inventory_sense_count(
            dst: List[List[WSDInstance]], inv_name: str
        ) -> None:
            curr_inventory = inventory2sense_count[inv_name]
            for sentence in dst:
                for instance in sentence:
                    if instance.labels is None:
                        continue
                    curr_inventory[instance.labels[0]] += 1

        for dataset, inventory_name in zip(self.datasets, self.datasets_inventory):
            update_inventory_sense_count(dataset, inventory_name)

        for sense_count in inventory2sense_count.values():
            total_occ = sum(sense_count.values())
            for sense, count in sense_count.items():
                self.sense_inverse_frequencies[sense] = 1 / (
                    count / total_occ
                )  # inverse doc freq

    def _torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(
                inputs, already_has_special_tokens=True
            )
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        defined_token_id = self.tokenizer.convert_tokens_to_ids(self.defined_special_token)
        defined_token_mask = inputs == defined_token_id
        ignored_indices = torch.logical_and(~masked_indices, ~defined_token_mask)
        labels[ignored_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def _init_final_dataset(self) -> None:
        def pick_instance(snt: List[WSDInstance]) -> int:
            selectable_instances_idx = [
                idx for idx, inst in enumerate(snt) if inst.labels is not None
            ]

            if len(selectable_instances_idx) == 0:
                return -1

            instances_inverse_frequency = np.array(
                [
                    self.sense_inverse_frequencies[snt[inst_idx].labels[0]]
                    for inst_idx in selectable_instances_idx
                ]
            )
            instances_probs = instances_inverse_frequency / np.sum(
                instances_inverse_frequency
            )
            return np.random.choice(selectable_instances_idx, p=instances_probs)

        def encode_and_apply_masking(
            snt: List[WSDInstance], instance_idx: int, definition: str
        ) -> Dict[str, Any]:
            input_words = " ".join([inst.annotated_token.text for inst in snt])
            tokenization_output = self.tokenizer(
                [input_words, definition], add_special_tokens=False
            )
            final_indices = [self.tokenizer.cls_token_id]
            final_indices += tokenization_output.input_ids[0]
            final_indices += [
                self.tokenizer.convert_tokens_to_ids(self.definition_special_token)
            ]
            final_indices += tokenization_output.input_ids[1]
            final_indices += [self.tokenizer.sep_token_id]
            final_indices = torch.tensor(final_indices, dtype=torch.long)

            input_indices = torch.clone(final_indices)
            defined_token_boundaries = [
                x + 1
                for x in tokenization_output.word_to_tokens(
                    instance_idx
                )  # + 1 for the cls
            ]
            input_indices[
                defined_token_boundaries[0] : defined_token_boundaries[1]
            ] = self.tokenizer.convert_tokens_to_ids(self.defined_special_token)

            inputs, labels = self._torch_mask_tokens(input_indices, final_indices)

            return {
                "input_ids": inputs,
                "attention_mask": torch.ones_like(inputs),
                "labels": labels,
                "sense": snt[instance_idx].labels[0],
            }

        print("Materializing final dataset...")
        for dataset, inventory_name in tqdm(
            zip(self.datasets, self.datasets_inventory)
        ):
            inventory = self.inventories[inventory_name]
            for sentence in tqdm(dataset, desc=f"Processing samples"):
                chosen_instance = pick_instance(sentence)
                if chosen_instance < 0:
                    continue
                instance_definition = inventory[sentence[chosen_instance].labels[0]]
                encoding_output = encode_and_apply_masking(
                    sentence, chosen_instance, instance_definition
                )
                self.final_dataset.append(encoding_output)


def main():
    dmlm_dataset = DMLMDataset(
        {"/home/edobobo/PycharmProjects/dmlm/data/datasets/oxford/oxf.0": "oxford"},
        {"oxford": "/home/edobobo/PycharmProjects/dmlm/data/inventories/oxf.tsv"},
        "bert-base-cased",
        "[DEF]",
        "[DEFINITION]",
        0.15,
    )

    senses_counter = Counter()
    for sample in dmlm_dataset:
        senses_counter[sample["sense"]] += 1

    print(senses_counter.most_common(10))

    sample = dmlm_dataset[100]
    print(sample)


if __name__ == "__main__":
    main()
