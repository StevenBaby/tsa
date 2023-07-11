import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch  # noqa
import torch.nn as nn  # noqa
import torch.cuda  # noqa
import torch.nn.functional as F  # noqa
import torch.utils.data  # noqa
import torchvision
import torchvision.transforms as T

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler


class TimeSeriesDataset(torch.utils.data.Dataset):

    def __init__(self, data, lags) -> None:
        super().__init__()

        self.lags = lags
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return (self.data.numel() - self.lags - 1)

    def __getitem__(self, index):
        return (
            self.data[index: self.lags + index],
            self.data[self.lags + index],
        )


class TrafficDataset(TimeSeriesDataset):

    def __init__(self, lags, train=True, len=None) -> None:
        if train:
            filename = 'datasets/train.csv'
        else:
            filename = 'datasets/test.csv'

        data = pd.read_csv(filename).fillna(0)
        data = data["Lane 1 Flow (Veh/5 Minutes)"].values[:len]
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

        super().__init__(data, lags)


class MLP(nn.Module):

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),

            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),

            nn.LazyLinear(1),
            nn.Sigmoid()
        )


class CNN(nn.Module):

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Unflatten(1, (1, -1)),

            nn.Conv1d(1, 64, 4, 2, 1),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),

            nn.Conv1d(64, 16, 4, 2, 1),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),

            nn.Flatten(),

            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),

            nn.LazyLinear(1),
            nn.Sigmoid()
        )


class LSTM(nn.Module):

    def forward(self, x: torch.Tensor):
        y = self.input(x)
        y, _ = self.rnn(y, None)
        y = self.output(y)
        return y

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input = nn.Sequential(
            nn.LazyLinear(16),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),
        )

        self.rnn = nn.LSTM(
            input_size=16,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid()
        )


class GRU(nn.Module):

    def forward(self, x: torch.Tensor):
        y = self.input(x)
        y, _ = self.rnn(y, None)
        y = self.output(y)
        return y

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input = nn.Sequential(
            nn.LazyLinear(16),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),
        )

        self.rnn = nn.GRU(
            input_size=16,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.LazyLinear(1),
            nn.Sigmoid()
        )


class SATT(nn.Module):

    def forward(self, x: torch.Tensor):
        y = self.input(x)

        m = torch.ones((x.shape[0], x.shape[1], 16))

        y, _ = self.attention(y, y, y)

        y = self.output(y)
        return y

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input = nn.Sequential(
            nn.LazyLinear(16 * 16),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),

            nn.Unflatten(1, (16, -1))
        )

        self.attention = nn.MultiheadAttention(16, 8)

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1),
            nn.Sigmoid()
        )
