import collections

from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from src.dmlm_dataset import SenseInventory, EfficientDMLMDataset


if __name__ == "__main__":

    oxford_inventory = SenseInventory(
        "/home/edobobo/PycharmProjects/dmlm/data/inventories/oxf.tsv"
    )

    witkionary_inventory = SenseInventory(
        "/home/edobobo/PycharmProjects/dmlm/data/inventories/wikt.tsv"
    )

    wordnet_inventory = SenseInventory(
        "/home/edobobo/PycharmProjects/dmlm/data/inventories/wn.tsv"
    )

    dmlm_dataset = EfficientDMLMDataset(
        "data/processed_datasets/debuggin.jsonl",
        {
            "oxford": oxford_inventory,
            "wiktionary": witkionary_inventory,
            "wordnet": wordnet_inventory,
        },
        "bert-base-cased",
        "[DEF]",
        "[DEFINITION]",
        0.15,
        plain_mlm_probability=0.0,
    )

    dataloader = DataLoader(
        dmlm_dataset,
        batch_size=128,
        num_workers=16,
        collate_fn=dmlm_dataset.collate_function,
    )

    senses_counter = collections.Counter()

    epochs = 1

    for _ in range(epochs):

        for batch in tqdm(dataloader):
            senses = [_s for _s in batch["senses"] if _s is not None]
            senses_counter.update(senses)

    print("Number of epochs:", epochs)
    print("Total number of senses:", len(senses_counter))
    print(
        "AVG occurrences per sense:",
        sum(senses_counter.values()) / len(senses_counter),
    )

    print(
        f"Senses occurrences percentiles:",
        np.percentile(list(senses_counter.values()), [25.0, 50.0, 75.0, 100.0]),
    )

    t = dmlm_dataset.tokenizer
    for i in range(len(batch["input_ids"])):
        composition_string = ""
        for j in range(len(batch["input_ids"][i])):
            label = batch["labels"][i][j]
            label = label if label > 0 else t.pad_token_id
            composition_string += (
                f"{t.decode(batch['input_ids'][i][j])} - {t.decode(label)} | "
            )
        print(composition_string)
        print()
