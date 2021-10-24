from torch import Tensor, distributions

from hw_asr.augmentations.base import AugmentationBase


class Noisify(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.noiser = distributions.Normal(0, 0.001)

    def __call__(self, data: Tensor):
        return data + self.noiser.sample(data.size())
