from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

from torch.utils.data.dataset import T_co
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, BatchEncoding

from src.utils.nns import batchify
from src.utils.wsd import WSDInstance, expand_raganato_path, read_from_raganato


class SenseInventory:
    def __init__(self, inventory_path: str):
        self.inventory_store = dict()
        self.load_inventory(inventory_path)

    def load_inventory(self, inv_path: str) -> None:
        with open(inv_path) as f:
            for line in f:
                sense_key, _, _, definition = line.strip().split("\t")
                self.inventory_store[sense_key] = definition

    def __getitem__(self, item):
        return self.inventory_store[item]


class MLMDataset(Dataset):
    def __init__(
        self,
        transformer_model: str,
        mlm_probability: float,
        additional_special_tokens: Optional[List[str]] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model,
            additional_special_tokens=additional_special_tokens,
        )
        self.mlm_probability = mlm_probability

    def collate_function(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return dict(
            input_ids=batchify(
                [sample["input_ids"] for sample in samples],
                padding_value=self.tokenizer.pad_token_id,
            ),
            attention_mask=batchify(
                [sample["attention_mask"] for sample in samples],
                padding_value=0,
            ),
            labels=batchify(
                [sample["labels"] for sample in samples],
                padding_value=-100,
            ),
        )

    def _torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor],
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if labels is None:
            labels = torch.clone(inputs)

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

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

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

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError


class BaseMLMDataset(MLMDataset):
    def __init__(
        self,
        datasets_path: List[str],
        transformer_model: str,
        mlm_probability: float,
        limit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(transformer_model, mlm_probability)
        self.dataset_store: List[str] = []

        self.final_dataset: List[Dict[str, Any]] = []

        self._load_data(datasets_path, limit)
        self.init_final_dataset()

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.final_dataset[item]

    def __len__(self):
        return len(self.final_dataset)

    def _load_data(self, datasets_path: List[str], limit: Optional[int] = None) -> None:
        for data_path in datasets_path:
            raganato_dataset = read_from_raganato(data_path)
            for _, _, wsd_sentence in raganato_dataset:
                if any(wi.annotated_token.text is None for wi in wsd_sentence):
                    continue
                self.dataset_store.append(
                    " ".join([wi.annotated_token.text for wi in wsd_sentence])
                )
                if limit is not None and len(self.final_dataset) == limit:
                    break

    def init_final_dataset(self) -> None:
        tokenized_sentences = self.tokenizer(self.dataset_store)
        for sentence_idx in range(len(self.dataset_store)):
            input_ids = torch.tensor(
                tokenized_sentences.input_ids[sentence_idx], dtype=torch.long
            )
            if len(input_ids) >= self.tokenizer.model_max_length:
                continue
            input_ids, labels = self._torch_mask_tokens(input_ids, labels=None)
            self.final_dataset.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": labels,
                }
            )
        print("Total instances in the dataset: ", len(self.final_dataset))


class DMLMDataset(MLMDataset):
    def __init__(
        self,
        inventory2datasets: Dict[str, List[str]],
        inventories: Dict[str, SenseInventory],
        transformer_model: str,
        defined_special_token: str,
        definition_special_token: str,
        mlm_probability: float,
    ):
        super().__init__(
            transformer_model,
            mlm_probability,
            additional_special_tokens=[defined_special_token, definition_special_token],
        )
        self.datasets: List[List[List[WSDInstance]]] = []
        self.datasets_inventory: List[str] = []
        self.inventories: Dict[str, SenseInventory] = inventories
        self.sense_inverse_frequencies: Dict[str, float] = dict()

        self.defined_special_token = defined_special_token
        self.definition_special_token = definition_special_token

        self.final_dataset: List[Dict[str, Any]] = []

        self._init_datasets(inventory2datasets)
        self._init_sense_inverse_frequencies()
        self.init_final_dataset()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.final_dataset[index]

    def _init_datasets(self, inventory2datasets: Dict[str, List[str]]) -> None:
        def load_dataset(dst_path: str) -> List[List[WSDInstance]]:
            return [x[-1] for x in read_from_raganato(*expand_raganato_path(dst_path))]

        print("Initializing datasets...")
        for inventory_name, inventory_datasets in inventory2datasets.items():
            for dataset_path in inventory_datasets:
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

    def _clean_dataset(
        self, dataset: List[List[WSDInstance]]
    ) -> List[List[WSDInstance]]:
        return [
            snt
            for snt in dataset
            if all([wi.annotated_token.text is not None for wi in snt])
        ]

    def _pretokenize_dataset(self, dataset: List[List[WSDInstance]]) -> BatchEncoding:
        dataset_sentences = [
            " ".join([wi.annotated_token.text for wi in snt]) for snt in dataset
        ]
        return self.tokenizer(dataset_sentences, add_special_tokens=False)

    @lru_cache(maxsize=100_000)
    def _tokenize_definition(self, definition: str) -> List[int]:
        return self.tokenizer(definition, add_special_tokens=False).input_ids

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
        defined_token_id = self.tokenizer.convert_tokens_to_ids(
            self.defined_special_token
        )
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

    def init_final_dataset(self) -> None:
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
            snt: List[WSDInstance],
            instance_idx: int,
            definition: str,
            sentence_idx: int,
        ) -> Dict[str, Any]:
            final_indices = (
                # [CLS]
                [self.tokenizer.cls_token_id]
                # sentence ids
                + tokenized_sentences.input_ids[sentence_idx]
                # [DEFINITION]
                + [self.tokenizer.convert_tokens_to_ids(self.definition_special_token)]
                # definition ids
                + self._tokenize_definition(definition)
                # [SEP]
                + [self.tokenizer.sep_token_id]
            )

            final_indices = torch.tensor(final_indices, dtype=torch.long)

            input_indices = torch.clone(final_indices)
            defined_token_boundaries = [
                x + 1
                for x in tokenized_sentences.word_to_tokens(
                    sentence_idx, instance_idx
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
            dataset = self._clean_dataset(dataset)
            tokenized_sentences = self._pretokenize_dataset(dataset)
            for i, sentence in tqdm(enumerate(dataset), desc=f"Processing samples"):
                chosen_instance = pick_instance(sentence)
                if chosen_instance < 0:
                    continue
                instance_definition = inventory[sentence[chosen_instance].labels[0]]
                encoding_output = encode_and_apply_masking(
                    sentence,
                    chosen_instance,
                    instance_definition,
                    i,
                )

                if len(encoding_output["input_ids"]) >= self.tokenizer.model_max_length:
                    continue

                self.final_dataset.append(encoding_output)

        print("Total instances in the dataset: ", len(self.final_dataset))

    def collate_function(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        return dict(
            input_ids=batchify(
                [sample["input_ids"] for sample in samples],
                padding_value=self.tokenizer.pad_token_id,
            ),
            attention_mask=batchify(
                [sample["attention_mask"] for sample in samples],
                padding_value=0,
            ),
            labels=batchify(
                [sample["labels"] for sample in samples],
                padding_value=-100,
            ),
        )

    def __len__(self) -> int:
        return len(self.final_dataset)


def main():

    oxford_inventory = SenseInventory(
        "/home/edobobo/PycharmProjects/dmlm/data/inventories/oxf.tsv"
    )

    dmlm_dataset = DMLMDataset(
        {"oxford": ["/home/edobobo/PycharmProjects/dmlm/data/datasets/oxford/oxf.0"]},
        {"oxford": oxford_inventory},
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
