#!/bin/bash


PYTHONPATH=. python src/scripts/convert_raganato_to_jsonl.py \
  --raganato-paths \
    data/datasets/oxford/oxf.10 \
    data/datasets/oxford/oxf.30 \
    data/datasets/oxford/oxf.50 \
    data/datasets/dmlm_datasets/oxf.11 \
    data/datasets/dmlm_datasets/oxf.12 \
    data/datasets/dmlm_datasets/oxf.13 \
    data/datasets/dmlm_datasets/oxf.31 \
    data/datasets/dmlm_datasets/oxf.32 \
    data/datasets/dmlm_datasets/oxf.33 \
    data/datasets/dmlm_datasets/oxf.51 \
    data/datasets/dmlm_datasets/oxf.52 \
    data/datasets/dmlm_datasets/oxf.53 \
    data/datasets/wiktionary/wikt.10 \
    data/datasets/wiktionary/wikt.30 \
    data/datasets/wiktionary/wikt.50 \
    data/datasets/dmlm_datasets/wikt.11 \
    data/datasets/dmlm_datasets/wikt.12 \
    data/datasets/dmlm_datasets/wikt.13 \
    data/datasets/dmlm_datasets/wikt.31 \
    data/datasets/dmlm_datasets/wikt.32 \
    data/datasets/dmlm_datasets/wikt.33 \
    data/datasets/dmlm_datasets/wikt.51 \
    data/datasets/dmlm_datasets/wikt.52 \
    data/datasets/dmlm_datasets/wikt.53 \
    data/datasets/wordnet/wn.10 \
    data/datasets/wordnet/wn.30 \
    data/datasets/wordnet/wn.50 \
    data/datasets/dmlm_datasets/wn.11 \
    data/datasets/dmlm_datasets/wn.12 \
    data/datasets/dmlm_datasets/wn.13 \
    data/datasets/dmlm_datasets/wn.31 \
    data/datasets/dmlm_datasets/wn.32 \
    data/datasets/dmlm_datasets/wn.33 \
    data/datasets/dmlm_datasets/wn.51 \
    data/datasets/dmlm_datasets/wn.52 \
    data/datasets/dmlm_datasets/wn.53 \
  --identifiers \
    oxford \
    oxford \
    oxford \
    oxford \
    oxford \
    oxford \
    oxford \
    oxford \
    oxford \
    oxford \
    oxford \
    oxford \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wiktionary \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
    wordnet \
  --output-path data/processed_datasets/dataset-3.5M.jsonl