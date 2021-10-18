import torch
from torch import nn
from torch.nn import Sequential
from torch.nn import functional as F

from hw_asr.base import BaseModel


class ConvBatchnormModel(BaseModel):
    def __init__(self, n_feats, n_class, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)

        self.bn1 = nn.BatchNorm1d(128)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=8, padding='same')
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=8, padding='same')
        self.conv3 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=8, padding='same')
        self.bn2 = nn.BatchNorm1d(256)

        self.lstm1 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm5 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=256, out_features=n_class)

    def forward(self, spectrogram, *args, **kwargs):
        x = spectrogram

        batch_size = x.shape[0]
        spect_size = x.shape[1]

        x = x.reshape(batch_size*spect_size, 128)
        x = self.bn1(x)
        x = x.reshape(batch_size*spect_size, 1, 128)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(batch_size*spect_size, 256)
        x = self.bn2(x)
        x = x.reshape(batch_size, spect_size, 256)

        
        x = F.relu(self.lstm1(x)[0])
        x = F.relu(self.lstm2(x)[0])
        x = F.relu(self.lstm3(x)[0])
        x = F.relu(self.lstm4(x)[0])
        x = F.relu(self.lstm5(x)[0])
        x = self.fc(x)

        return x

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
