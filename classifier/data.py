# -*- coding: utf-8 -*-
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, features, targets):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class Data(object):
    def __init__(self):
        pass

    def load_from_file(self, filename):
        pass

    def process(self, raw_data):
        pass

    def get_dataset(self):
        pass
