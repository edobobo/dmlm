import math
from abc import ABC

from typing import Iterator, Sized, Sequence, List, Optional

import numpy as np

from torch.utils.data import Sampler

from src.utils.commons import chunks, flatten


class NoisedBatchSampler(Sampler[Sequence[int]], ABC):
    def __init__(self, data_source: Sized, noise_value: float):
        super().__init__(data_source)
        self.sizes = np.asarray(data_source)
        self.noise_value = noise_value
        self.noisy_indices = self._get_noisy_indices()

    def _get_noisy_indices(self):
        return self.noisy_argsort(self.sizes, self.noise_value)

    @classmethod
    def noised(cls, array: np.ndarray, noise_value: float = 0.05) -> np.ndarray:
        noise = 1 + np.random.uniform(-noise_value, noise_value, array.shape)
        noised_array = array * noise
        return noised_array

    @classmethod
    def noisy_argsort(cls, array: np.ndarray, noise_value: float = 0.05) -> np.ndarray:
        sorted_indices = cls.noised(array, noise_value).argsort()
        sorted_chunks = list(chunks(sorted_indices, 4096))
        np.random.shuffle(sorted_chunks)
        return np.array(flatten(sorted_chunks))


class PadReducerBatchSampler(NoisedBatchSampler):
    def __init__(self, data_source: Sized, batch_size: int, noise_value: float = 0.05):
        super().__init__(data_source, noise_value)
        self.sizes = np.array(data_source, dtype=np.int32)
        self.batch_size = batch_size
        self._num_batches = math.ceil(len(self.sizes) / batch_size)
        self._batches_indices = np.arange(self._num_batches) * self.batch_size

    def __iter__(self) -> Iterator[Sequence[int]]:
        np.random.shuffle(self._batches_indices)

        for batch_index_start in self._batches_indices:
            batch_index_end = batch_index_start + self.batch_size
            yield self.noisy_indices[batch_index_start:batch_index_end]

    def __len__(self):
        return self._num_batches


class MaxTokensBatchSampler(NoisedBatchSampler):
    def __init__(self, data_source: Sized, max_tokens: int, max_batch_size: Optional[int] = None, noise_value: float = 0.05):
        super().__init__(data_source, noise_value)
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.batch_indicators = self._get_batches()

    def _lazy_groups_of_max_size(self):
        cur_max_size = 0
        starting_index = 0
        group: List[int] = []
        max_size = self.max_tokens

        for i, index in enumerate(self.noisy_indices):
            size = self.sizes[index]

            # if size > self.max_tokens:
            #     logger.warning(
            #         "Found instance of size %d, which is bigger than the expected size for a batch (%d)",
            #         size,
            #         self.max_tokens,
            #     )
            group_size = max(size, cur_max_size) * (len(group) + 1)

            if group_size > max_size or (self.max_batch_size is not None and len(group) >= self.max_batch_size):
                yield starting_index, len(group)
                cur_max_size = 0
                group = []
                starting_index = i

            group.append(index)
            cur_max_size = max(cur_max_size, size)

        if len(group) > 0:
            yield starting_index, len(group)

    def _get_batches(self):
        batch_indicators = list(self._lazy_groups_of_max_size())
        np.random.shuffle(batch_indicators)
        return batch_indicators

    def __iter__(self) -> Iterator[Sequence[int]]:
        for starting_index, batch_length in self.batch_indicators:
            yield self.noisy_indices[
                starting_index : starting_index + batch_length
            ].tolist()

        self.batch_indicators = self._get_batches()

    def __len__(self):
        return len(self.batch_indicators)


if __name__ == "__main__":
    import random
    from time import time

    from datasets import load_from_disk, concatenate_datasets
    from tqdm.auto import tqdm

    start = time()

    ds = [
        load_from_disk(path)
        for path in [
            "data/processed/bookcorpus/bert-base-cased-group@128-doc_broken",
            "data/processed/wikisents/en/bert-base-cased-group@128-doc_broken",
        ]
    ]

    print(f"Loading the dataset took {time() - start:.3f} seconds")
    start = time()

    splits = {split for d in ds for split in d.keys()}
    datasets = {split: concatenate_datasets([d[split] for d in ds]) for split in splits}
    print(f"Concatenating the datasets took {time() - start:.3f} seconds")

    # random.seed(42)
    # torch.random.manual_seed(42)

    # sizes = torch.randint(1, 128, (1000000,)).tolist()

    start = time()
    sizes = datasets["train"]["length"]
    print(f"Loading lengths took {time() - start:.3f} seconds")
    sizes_np = np.array(sizes)
    # sampler = BatchSampler(RandomSampler(sizes), batch_size=128, drop_last=False)
    # sampler = PadReducerBatchSampler(sizes, batch_size=128)
    sampler = MaxTokensBatchSampler(sizes, max_tokens=8192, noise_value=0.05)
    all_indices = set(list(range(len(sizes))))
    pad_wasted = []

    for i, batch_indices in enumerate(tqdm(sampler)):
        elements = sizes_np[batch_indices].tolist()
        all_indices.difference_update(batch_indices)
        tensor_size = len(elements) * max(elements)
        tokens_size = sum(elements)
        padding_ratio = 1 - tokens_size / tensor_size
        pad_wasted.append(padding_ratio)

    print(
        f"We have a total of {len(sampler)} batches ({len(all_indices)} indices remaining)"
    )
    print(f"Wasted {np.mean(pad_wasted) * 100:.4f}% on average on padding")
