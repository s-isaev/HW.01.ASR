import torch
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class PetConvBatchnormModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.norm1 = nn.BatchNorm1d(128)
        self.conv = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=8, padding='same')
        self.norm2 = nn.BatchNorm1d(256)
        self.lstm  = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional = True)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        x = spectrogram

        batc_size = x.shape[0]
        spec_size = x.shape[1]
        feat_size = x.shape[2]

        x = x.reshape(batc_size*spec_size, feat_size)
        x = self.norm1(x)
        x = x.reshape(batc_size, spec_size, feat_size)

        x = x.reshape(batc_size*spec_size, 1, feat_size)
        x = self.conv(x)
        x = x.reshape(batc_size, spec_size, 2*feat_size)

        x = x.reshape(batc_size*spec_size, 2*feat_size)
        x = self.norm2(x)
        x = x.reshape(batc_size, spec_size, 2*feat_size)

        x, (hn, cn) = self.lstm(x)
        return self.net(x)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
