import torch.nn as nn


class OneLayerPerceptron(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_classes):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
        self.linear = nn.Linear(in_features=embedding_dim * 50, out_features=num_classes)

    def forward(self, x):
        x = self.embeddings(x)
        bsz = x.shape[0]
        x = x.view(bsz, -1)
        x = self.linear(x)
        return x
