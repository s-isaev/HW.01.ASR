import librosa, random
from torch import Tensor
import torch

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: Tensor):
        c = random.uniform(0.8, 1.2)
        data = librosa.effects.time_stretch(data.numpy().squeeze(), c)
        data = torch.from_numpy(data).unsqueeze(0)
        return data
