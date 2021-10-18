import torch
from torch import nn
from torch.nn import Sequential
from torch.nn import functional as F

from hw_asr.base import BaseModel


class ConvBatchnormModel(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.lstm1  = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.relu1 = nn.ReLU()
        self.lstm2  = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.relu2 = nn.ReLU()
        self.lstm3  = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.relu3 = nn.ReLU()
        self.lstm4  = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.relu4 = nn.ReLU()
        self.lstm5  = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.relu5 = nn.ReLU()
        self.linear = nn.Linear(in_features=256, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        x, (hn, cn) = self.lstm1(spectrogram)
        x = self.relu1(x)
        x, (hn, cn) = self.lstm2(x)
        x = self.relu2(x)
        x, (hn, cn) = self.lstm3(x)
        x = self.relu3(x)
        x, (hn, cn) = self.lstm4(x)
        x = self.relu4(x)
        x, (hn, cn) = self.lstm5(x)
        x = self.relu5(x)
        return self.linear(x)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
