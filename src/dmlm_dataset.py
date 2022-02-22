from collections import Counter, defaultdict
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple

import datasets
from datasets import Features
from torch.utils.data.dataset import T_co
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, BatchEncoding

from src.utils.nns import batchify
from src.utils.wsd import WSDInstance, expand_raganato_path, read_from_raganato

import logging


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

        self.logger = logging.getLogger(BaseMLMDataset.__name__)

        self.dataset_store: List[str] = []

        self.final_dataset: List[Dict[str, Any]] = []

        self._load_data(datasets_path, limit)
        self.init_final_dataset()

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.final_dataset[item]

    def __len__(self):
        return len(self.final_dataset)

    def _load_data(self, datasets_path: List[str], limit: Optional[int] = None) -> None:
        self.logger.info("Loading datasets from raganato files...")
        for data_path in datasets_path:
            self.logger.info(f"Loading from: {data_path}")
            raganato_dataset = read_from_raganato(data_path)
            for _, _, wsd_sentence in raganato_dataset:
                if any(wi.annotated_token.text is None for wi in wsd_sentence):
                    continue
                self.dataset_store.append(
                    " ".join([wi.annotated_token.text for wi in wsd_sentence])
                )
                if limit is not None and len(self.dataset_store) == limit:
                    break

    def init_final_dataset(self) -> None:
        self.logger.info("Initializing final dataset")
        self.final_dataset = []
        tokenized_sentences = self.tokenizer(self.dataset_store)
        for sentence_idx in tqdm(range(len(self.dataset_store))):
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
        self.logger.info(f"Total instances in the dataset: {len(self.final_dataset)}")


class DMLMDataset(MLMDataset):
    def __init__(
        self,
        inventory2datasets: Dict[str, List[str]],
        inventories: Dict[str, SenseInventory],
        transformer_model: str,
        defined_special_token: str,
        definition_special_token: str,
        mlm_probability: float,
        plain_mlm_probability: float = 0.0,
    ):
        super().__init__(
            transformer_model,
            mlm_probability,
            additional_special_tokens=[defined_special_token, definition_special_token],
        )

        self.plain_mlm_probability = plain_mlm_probability

        self.logger = logging.getLogger(DMLMDataset.__name__)

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

        self.logger.info("Initializing datasets...")
        for inventory_name, inventory_datasets in inventory2datasets.items():
            for dataset_path in inventory_datasets:
                self.logger.info(f"Loading from: {dataset_path}")
                self.datasets.append(load_dataset(dataset_path))
                self.datasets_inventory.append(inventory_name)

    def _init_sense_inverse_frequencies(self) -> None:
        self.logger.info("Computing senses inverse frequencies")
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
            special_tokens_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(
                        _input, already_has_special_tokens=True
                    )
                    for _input in inputs
                ],
                dtype=torch.bool,
            )
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
            instance_idx: Optional[int] = None,
            definition: Optional[str] = None,
            sentence_idx: Optional[int] = None,
        ) -> Dict[str, Any]:
            if instance_idx is not None:
                final_indices = (
                    # [CLS]
                    [self.tokenizer.cls_token_id]
                    # sentence ids
                    + tokenized_sentences.input_ids[sentence_idx]
                    # [DEFINITION]
                    + [
                        self.tokenizer.convert_tokens_to_ids(
                            self.definition_special_token
                        )
                    ]
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

            else:
                input_indices = (
                    # [CLS]
                    [self.tokenizer.cls_token_id]
                    # sentence ids
                    + tokenized_sentences.input_ids[sentence_idx]
                    # [SEP]
                    + [self.tokenizer.sep_token_id]
                )
                input_indices = torch.tensor(input_indices, dtype=torch.long)
                final_indices = input_indices.clone()

            return {
                "input_indices": input_indices,
                "final_indices": final_indices,
                "sense": snt[instance_idx].labels[0]
                if instance_idx is not None
                else None,
            }

        self.logger.info("Materializing final dataset...")
        self.final_dataset = []
        for dataset, inventory_name in tqdm(
            zip(self.datasets, self.datasets_inventory)
        ):
            inventory = self.inventories[inventory_name]
            dataset = self._clean_dataset(dataset)
            tokenized_sentences = self._pretokenize_dataset(dataset)

            if self.plain_mlm_probability > 0:
                is_plain_mlm = torch.bernoulli(
                    torch.full((len(dataset),), self.plain_mlm_probability)
                ).bool()
            else:
                is_plain_mlm = [False] * len(dataset)

            for i, (sentence, plain_mlm) in tqdm(
                enumerate(zip(dataset, is_plain_mlm)), desc=f"Processing samples"
            ):
                if plain_mlm:
                    encoding_output = encode_and_apply_masking(sentence, sentence_idx=i)
                else:
                    chosen_instance = pick_instance(sentence)
                    if chosen_instance < 0:
                        continue
                    instance_definition = inventory[sentence[chosen_instance].labels[0]]
                    encoding_output = encode_and_apply_masking(
                        sentence,
                        chosen_instance,
                        instance_definition,
                        sentence_idx=i,
                    )

                if (
                    len(encoding_output["input_indices"])
                    >= self.tokenizer.model_max_length
                ):
                    continue

                self.final_dataset.append(encoding_output)

        self.logger.info(f"Total instances in the dataset: {len(self.final_dataset)}")

    def collate_function(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:

        input_indices = batchify(
            [sample["input_indices"] for sample in samples],
            padding_value=self.tokenizer.pad_token_id,
        )

        attention_mask = batchify(
            [torch.ones_like(sample["input_indices"]) for sample in samples],
            padding_value=0,
        )

        final_indices = batchify(
            [sample["final_indices"] for sample in samples],
            padding_value=self.tokenizer.pad_token_id,
        )

        input_ids, labels = self._torch_mask_tokens(input_indices, final_indices)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def __len__(self) -> int:
        return len(self.final_dataset)


def offsets_to_word2tokens(example):
    word2tokens = list()
    first_start = -1
    last_end = -1
    last_bpe_inserted = 0
    for i, (start, end) in enumerate(example["offset_mapping"]):

        if first_start == -1:
            first_start = start
            last_end = end
        elif start == last_end:
            last_end = end
        else:
            word2tokens.append((last_bpe_inserted, i))
            last_bpe_inserted = i
            first_start = start
            last_end = end

        if i == len(example["offset_mapping"]) - 1:
            word2tokens.append((last_bpe_inserted, i + 1))

    example["word2tokens"] = word2tokens

    return example


class EfficientDMLMDataset(MLMDataset):
    def __init__(
        self,
        dataset_path: str,
        inventories: Dict[str, SenseInventory],
        transformer_model: str,
        defined_special_token: str,
        definition_special_token: str,
        mlm_probability: float,
        plain_mlm_probability: float = 0.0,
    ):
        super().__init__(
            transformer_model,
            mlm_probability,
            additional_special_tokens=[defined_special_token, definition_special_token],
        )

        self.logger = logging.getLogger(EfficientDMLMDataset.__name__)

        self.defined_special_token = defined_special_token
        self.definition_special_token = definition_special_token

        self.dataset_path = dataset_path
        self.inventories = inventories
        self.plain_mlm_probability = plain_mlm_probability

        self.dataset_store = None
        self.sense_inverse_frequencies = None
        self.lengths = None

        self._load_dataset()
        self._clean_dataset()
        self._tokenize_dataset()
        self._compute_lengths()
        self._load_sense_inverse_frequencies()

    def _load_dataset(self):
        self.logger.info("Loading dataset")
        self.dataset_store = datasets.load_dataset(
            "json",
            data_files=self.dataset_path,
            features=Features(
                {
                    "sentence": datasets.Value("string"),
                    "labels": datasets.Sequence(datasets.Value("string")),
                    "dataset_id": datasets.Value("string"),
                }
            ),
        )["train"]

    def _clean_dataset(self):
        self.logger.info("Cleaning dataset")
        self.dataset_store = self.dataset_store.filter(
            lambda example: all(token is not None for token in example["sentence"])
        )

    def _tokenize_dataset(self):
        self.logger.info("Tokenizing dataset")
        self.dataset_store = self.dataset_store.map(
            lambda examples: self.tokenizer(
                examples["sentence"],
                return_token_type_ids=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
                add_special_tokens=False,
            ),
            batched=True,
        )

        self.dataset_store = self.dataset_store.filter(
            lambda examples: len(examples["input_ids"])
            < self.tokenizer.model_max_length
        )

        self.dataset_store = self.dataset_store.map(offsets_to_word2tokens)

    def _compute_lengths(self):
        self.dataset_store = self.dataset_store.map(
            lambda example: dict(length=len(example["input_ids"]))
        )
        self.lengths = self.dataset_store["length"]

    def _load_sense_inverse_frequencies(self):
        self.logger.info("Computing senses inverse frequencies")
        inventory2sense_count: Dict[str, Counter] = defaultdict(Counter)

        for sample in self.dataset_store:
            inventory2sense_count[sample["dataset_id"]].update(
                _l for _l in sample["labels"] if _l is not None
            )

        self.sense_inverse_frequencies = dict()
        for sense_count in inventory2sense_count.values():
            total_occ = sum(sense_count.values())
            for sense, count in sense_count.items():
                self.sense_inverse_frequencies[sense] = 1 / (
                    count / total_occ
                )  # inverse doc freq

    def __getitem__(self, item):
        return self.dataset_store[item]

    def __len__(self):
        return len(self.dataset_store)

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
            special_tokens_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(
                        _input, already_has_special_tokens=True
                    )
                    for _input in inputs
                ],
                dtype=torch.bool,
            )
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

    def pick_instance(self, labels: List[str]) -> int:
        selectable_instances_idx = [
            idx for idx, _label in enumerate(labels) if _label != "Nil"
        ]

        if len(selectable_instances_idx) == 0:
            return -1

        instances_inverse_frequency = np.array(
            [
                self.sense_inverse_frequencies[labels[inst_idx]]
                for inst_idx in selectable_instances_idx
            ]
        )
        instances_probs = instances_inverse_frequency / np.sum(
            instances_inverse_frequency
        )

        return np.random.choice(selectable_instances_idx, p=instances_probs)

    def encode_and_apply_masking(
        self,
        sample: Dict[str, Any],
        instance_idx: Optional[int] = None,
        definition: Optional[str] = None,
    ) -> Dict[str, Any]:
        if instance_idx is not None:
            final_indices = (
                # [CLS]
                [self.tokenizer.cls_token_id]
                # sentence ids
                + sample["input_ids"]
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
                x + 1 for x in sample["word2tokens"][instance_idx]
            ]
            input_indices[
                defined_token_boundaries[0] : defined_token_boundaries[1]
            ] = self.tokenizer.convert_tokens_to_ids(self.defined_special_token)

        else:
            input_indices = (
                # [CLS]
                [self.tokenizer.cls_token_id]
                # sentence ids
                + sample["input_ids"]
                # [SEP]
                + [self.tokenizer.sep_token_id]
            )
            input_indices = torch.tensor(input_indices, dtype=torch.long)
            final_indices = input_indices.clone()

        return {
            "input_indices": input_indices,
            "final_indices": final_indices,
        }

    def process_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_samples = []

        if self.plain_mlm_probability > 0:
            is_plain_mlm = torch.bernoulli(
                torch.full((len(samples),), self.plain_mlm_probability)
            ).bool()
        else:
            is_plain_mlm = [False] * len(samples)

        for plain_mlm, sample in zip(is_plain_mlm, samples):

            if plain_mlm:
                encoding_output = self.encode_and_apply_masking(sample)
            else:
                chosen_instance = self.pick_instance(sample["labels"])
                if chosen_instance < 0:
                    continue

                instance_definition = self.inventories[sample["dataset_id"]][
                    sample["labels"][chosen_instance]
                ]
                encoding_output = self.encode_and_apply_masking(
                    sample, chosen_instance, instance_definition
                )

                encoding_output["sense"] = sample["labels"][chosen_instance]

            processed_samples.append(encoding_output)

        return processed_samples

    def collate_function(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:

        processed_samples = self.process_samples(samples)

        input_indices = batchify(
            [ps["input_indices"] for ps in processed_samples],
            padding_value=self.tokenizer.pad_token_id,
        )

        attention_mask = batchify(
            [torch.ones_like(ps["input_indices"]) for ps in processed_samples],
            padding_value=0,
        )

        final_indices = batchify(
            [ps["final_indices"] for ps in processed_samples],
            padding_value=self.tokenizer.pad_token_id,
        )

        senses = [ps.get("sense", None) for ps in processed_samples]

        input_ids, labels = self._torch_mask_tokens(input_indices, final_indices)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            senses=senses,
        )

    def init_final_dataset(self) -> None:
        return


def main():

    oxford_inventory = SenseInventory(
        "/home/edobobo/PycharmProjects/dmlm/data/inventories/oxf.tsv"
    )

    dmlm_dataset = EfficientDMLMDataset(
        "data/processed_datasets/oxf_10.jsonl",
        {"oxford": oxford_inventory},
        "bert-base-cased",
        "[DEF]",
        "[DEFINITION]",
        0.15,
        plain_mlm_probability=0.5,
    )

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dmlm_dataset, batch_size=16, collate_fn=dmlm_dataset.collate_function
    )

    for batch in dataloader:
        print(batch)

    senses_counter = Counter()
    for sample in dmlm_dataset:
        senses_counter[sample["sense"]] += 1

    print(senses_counter.most_common(10))

    sample = dmlm_dataset[100]
    print(sample)


if __name__ == "__main__":
    main()
