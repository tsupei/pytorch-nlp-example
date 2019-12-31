# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Trainee(object):
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def set_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def save_model(self):
        pass

    def load_model(self):
        pass
