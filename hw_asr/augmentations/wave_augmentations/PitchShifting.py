import librosa, random
from torch import Tensor
import torch
from hw_asr.augmentations.base import AugmentationBase


class PitchShifting(AugmentationBase):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: Tensor):
        c = random.randint(-5, 5)
        data = librosa.effects.pitch_shift(data.numpy().squeeze(), 16_000, c)
        data = torch.from_numpy(data).unsqueeze(0)
        return data
