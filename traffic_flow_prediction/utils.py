import os
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
