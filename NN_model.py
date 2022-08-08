import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, num_units=128):
        super(NeuralNet, self).__init__()

        self.dense0 = nn.Linear(18, num_units)
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(num_units)
        self.dense_hidden = nn.Linear(num_units, num_units)
        self.dense1 = nn.Linear(num_units, 16)
        self.output = nn.Linear(16, 2)

    def forward(self, X, **kwargs):
        X = F.relu(self.dense0(X))
        X = self.dropout(X)
        X = self.dense_hidden(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = self.dense_hidden(X)
        X = F.relu(X)
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X