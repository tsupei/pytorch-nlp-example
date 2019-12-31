# -*- coding: utf-8 -*-
from torch.utils.data.dataset import Dataset
import torch
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.size = len(x)
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class TextData(object):
    """Turn Raw Text into data compatible with models"""
    def __init__(self, doc_size):
        self.x = []  # x represented as input
        self.y = []  # y represented as target
        self.target_lookup = []
        self.vocab_dict = self._load_vocab()
        self.doc_size = doc_size

    def _load_vocab(self):
        vocab_dict = {}
        with open("/Users/eyesmedia/Documents/lab/pytorch-nlp-example/classifier/vocab.txt", 'r', encoding="utf8") as file:
            txt = file.read()
            lines = txt.split('\n')
            lines = list(filter(lambda k: k, lines))
            for idx, line in enumerate(lines):
                vocab_dict[line] = idx
        return vocab_dict

    def text2index(self, text):
        """
        args:
            text (list): list of tokens
        return:
            indexed (list): list of index with fixed size
        """
        indexed = []
        for char in text:
            if char not in self.vocab_dict:
                indexed.append(self.vocab_dict["[UNK]"])
            else:
                indexed.append(self.vocab_dict[char])
        if len(indexed) < self.doc_size:
            indexed.extend([self.vocab_dict["[PAD]"]] * (self.doc_size-len(indexed)))
        else:
            indexed = indexed[:self.doc_size]
        return indexed

    def target2index(self, target):
        if target in self.target_lookup:
            return self.target_lookup.index(target)
        else:
            self.target_lookup.append(target)
            return len(self.target_lookup)-1

    def index2target(self, index):
        """
        args:
            index (int): index of target
        """
        if 0 <= index < len(self.target_lookup):
            return self.target_lookup[index]
        raise IndexError("{index} is not in the range of dictionary".format(index=index))

    def process(self, data):
        """
        args:
            data (list): list of data composed of (text, target)
        """
        with tqdm(total=len(data)) as progress_bar:
            for text, target in data:
                indexed = self.text2index(text)
                target = self.target2index(target)
                self.x.append(indexed)
                self.y.append(target)
                progress_bar.update(1)

    def get_dataset(self):
        return TextDataset(self.x, self.y)
