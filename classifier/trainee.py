# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from classifier.model.one_layer_perceptron import OneLayerPerceptron


class Trainee(object):
    def __init__(self):
        self.model = OneLayerPerceptron(num_embeddings=4840, embedding_dim=256, num_classes=4)

    def train(self, train_data, epoch, bsz, lr):
        data_loader = DataLoader(train_data.get_dataset(), num_workers=4, batch_size=bsz, drop_last=True, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        for i in range(epoch):
            with tqdm(total=len(data_loader)) as progress_bar:
                for train_x, train_y in data_loader:
                    optimizer.zero_grad()
                    logit = self.model(train_x)
                    loss = F.cross_entropy(logit, train_y)
                    tqdm.write("Loss: {}".format(loss.item()))
                    loss.backward()
                    optimizer.step()
                    progress_bar.update(1)

    def predict(self):
        pass

    def set_seed(self, seed):
        """Ensure reproducible model"""
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def save_model(self):
        pass

    def load_model(self):
        pass
