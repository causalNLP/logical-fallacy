import pickle
import cv2
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AdamW, AutoModelForSequenceClassification
import pandas as pd
import random
from torch import nn
import sklearn
import numpy as np
import sklearn.metrics
import logging
import argparse
from random import sample
from library import eval_classwise, eval_and_store, convert_to_multilabel, get_corefs, replace_masked_tokens, \
    replace_char
from logicedu import get_logger, MNLIDataset

device = "cpu"
torch.manual_seed(0)
logger = get_logger()
fallacy_train = pd.read_csv('../../data/edu_train.csv')
fallacy_dev = pd.read_csv('../../data/edu_dev.csv')
fallacy_test = pd.read_csv('../../data/edu_test.csv')
fallacy_ds = MNLIDataset('saved_models/electra-base-mnli', fallacy_train, fallacy_dev, 'updated_label',
                         'masked-logical-form',
                         fallacy_test,
                         fallacy=True, train_strat=3, test_strat=2)
model = AutoModelForSequenceClassification.from_pretrained('saved_models/electra-base-mnli', num_labels=3)
model.resize_token_embeddings(len(fallacy_ds.tokenizer))
logger.info("Generating results on Dev Set")
df = eval_and_store(fallacy_ds, model, logger, device, 'masked-logical-form', debug=True)
df = convert_to_multilabel(df)
df.to_csv('results/results.csv')
