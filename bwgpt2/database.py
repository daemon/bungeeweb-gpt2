from collections import defaultdict
from dataclasses import dataclass
from typing import List
import os
import random

from tqdm import tqdm
import pandas as pd


@dataclass
class ChatRecord(object):
    time: int
    uuid: str
    username: str
    content: str


@dataclass
class ChatLog(object):
    records: List[ChatRecord]
    dataframe: pd.DataFrame = None

    def __post_init__(self):
        if self.dataframe is not None:
            return
        data = defaultdict(list)
        for record in self.records:
            data['time'].append(record.time)
            data['uuid'].append(record.uuid.replace('\t', ''))
            data['username'].append(record.username.replace('\t', ' '))
            data['content'].append(record.content.replace('\t', ' ').replace('\n', ' '))
        self.dataframe = pd.DataFrame(data)

    def write_tsv(self, filename):
        self.dataframe.to_csv(filename, index=False, quoting=3, sep='\t', escapechar='\\')

    @classmethod
    def from_tsv(cls, filename, use_tqdm=True):
        df = pd.read_csv(filename, sep='\t', escapechar='\\', keep_default_na=False, quoting=3)[['time', 'uuid', 'username', 'content']]
        return cls.from_df(df, use_tqdm=use_tqdm)

    @classmethod
    def from_df(cls, df, use_tqdm=True):
        return cls([ChatRecord(*row[1:]) for row in tqdm(df.itertuples(), disable=not use_tqdm)], df)

    def transform(self, transform_fn):
        return self.from_df(transform_fn(self.dataframe), use_tqdm=False)
