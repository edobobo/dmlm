from pathlib import Path
from typing import Optional, Iterator, Sequence

import torch.distributed as dist
from more_itertools import chunked
from torch.utils.data import Sampler


class DistributedBatchSampler(Sampler[Sequence[int]]):
    def __init__(
        self,
        batch_sampler: Sampler[Sequence[int]],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        dump_batches: bool = False,
    ):

        super().__init__(None)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        print(f"Creating new DistributedBatchSampler at rank {rank}")
        self.sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.dump_batches = dump_batches
        self.epoch = 0

    def __iter__(self) -> Iterator[Sequence[int]]:
        iterator = chunked(self.sampler, self.num_replicas)

        out_dir = Path.cwd() / f'batch_dump/e{self.epoch}'
        out_dir.mkdir(exist_ok=True, parents=True)
        fd = (out_dir / f'{self.rank}.log').open('w') if self.dump_batches else None

        for replicas_batches in iterator:
            # skips the last batch
            if len(replicas_batches) != self.num_replicas:
                continue

            if fd is not None:
                print(replicas_batches, file=fd)

            # yields the batch corresponding to this rank
            yield replicas_batches[self.rank]

        if fd is not None:
            fd.close()

    def __len__(self) -> int:
        return len(self.sampler) // self.num_replicas

    # kept here just because it exists in DistributedSampler
    # and I'd like to avoid crashing at the end of the first epoch :)
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
