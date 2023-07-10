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
from sklearn import metrics


def sine_dataset(count=10000):
    x = np.linspace(0, 1000, count)
    y = np.abs(np.sin(x) + np.random.randn(x.shape[0]) * 0.03)
    return y


def load_sine_dataset(filename):
    if os.path.exists(filename):
        data = pd.read_csv(filename)
    else:
        raise IOError()
    return data


def save_sine_dataset(dataset, filename):
    if not isinstance(dataset, pd.DataFrame):
        dataset = pd.DataFrame(dataset)

    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    dataset.to_csv(filename, index=False)


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


class SineDataset(TimeSeriesDataset):

    def __init__(self, lags, train=True) -> None:
        if train:
            filename = 'datasets/sine.train.csv'
        else:
            filename = 'datasets/sine.test.csv'

        if not os.path.exists(filename):
            data = sine_dataset()
            save_sine_dataset(data, filename)
        data = load_sine_dataset(filename).values
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        data = self.scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)

        super().__init__(data, lags)
