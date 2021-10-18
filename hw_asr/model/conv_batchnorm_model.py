import torch
from torch import nn
from torch.nn import Sequential
from torch.nn import functional as F

from hw_asr.base import BaseModel


class ConvBatchnormModel(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.norm1 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=8, padding='same')
        self.crelu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=8, padding='same')
        self.crelu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=8, padding='same')
        self.crelu3 = nn.ReLU()
        self.norm2 = nn.BatchNorm1d(256)

        self.lstm1  = nn.LSTM(input_size=512, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
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
        x = spectrogram

        batc_size = x.shape[0]
        spec_size = x.shape[1]
        feat_size = x.shape[2]

        x = x.reshape(batc_size*spec_size, feat_size)
        x = self.norm1(x)
        x = x.reshape(batc_size*spec_size, 1, feat_size)
        x = self.conv1(x)
        x = self.crelu1(x)
        x = self.conv2(x)
        x = self.crelu2(x)
        x = self.conv3(x)
        x = self.crelu3(x)
        x = x.reshape(batc_size*spec_size, feat_size*4)
        x = self.norm2(x)
        x = x.reshape(batc_size, spec_size, feat_size*4)

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
