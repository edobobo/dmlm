#!/usr/bin/bash


PYTHONPATH=. python src train.py \
  train.model_name=bart-6-2-dmlm50%-fp16-dataset3.5M-lb \
  data=efficient_data_seq2seq \
  model=bart_seq2seq_model \
  data.train_dataset.dataset_path=data/preprocessed_datasets/dataset-3.5M
