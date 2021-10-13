import torch
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class OverfitModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        self.lstm  = nn.LSTM(input_size=n_feats, hidden_size=128, num_layers=2, batch_first=True, bidirectional = True)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_class)
        )

    def forward(self, spectrogram, *args, **kwargs):
        x, (hn, cn) = self.lstm(spectrogram)
        return self.net(x)

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
