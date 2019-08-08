from collections import Counter
from dataclasses import dataclass
from typing import List
import glob
import os
import random

from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch.utils.data as tud


@dataclass
class FlatFileDataset(tud.Dataset):
    sentences: List[str]

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)

    @classmethod
    def splits(cls, folder, train_file='train.txt', dev_file='dev.txt', test_file='test.txt'):
        sentences_lst = []
        for file_ in (train_file, dev_file, test_file):
            with open(os.path.join(folder, file_)) as f:
                sentences_lst.append([l.strip() for l in f.readlines()])
        return [cls(x) for x in sentences_lst]


def tokenize_batch(sentences,
                   tokenize_fn=list,
                   eos='<eos>',
                   pad='<pad>',
                   max_len=100,
                   pad_to_max=False):
    eos_append = [eos] if eos else []
    tokens_lst = [tokenize_fn(x) + eos_append for x in sentences]
    tokens_mask = [[1] * len(x) for x in tokens_lst]
    max_len = max_len if pad_to_max else min(max(map(len, tokens_lst)), max_len)
    tokens_lst = [x[:max_len] for x in tokens_lst]
    tokens_mask = [x[:max_len] for x in tokens_mask]
    tokens_lst = [x + ['<pad>'] * (max_len - len(x)) for x in tokens_lst]
    tokens_mask = [x + [0] * (max_len - len(x)) for x in tokens_mask]
    return tokens_lst, tokens_mask


def tokens_reduce(loss, tokens_lst):
    mask = []
    for tokens in tokens_lst:
        mask.append([int(x != '<pad>') for x in tokens])
    mask = torch.Tensor(mask)
    loss = loss * mask
    return loss.sum() / mask.sum()