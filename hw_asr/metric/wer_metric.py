from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], log_probs_length, *args, **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).tolist()
        lenghts = log_probs_length.tolist()
        for pred_vec, target_text, lenght in zip(predictions, text, lenghts):
            pred = pred_vec[:lenght]
            pred_text = self.text_encoder.ctc_decode(pred)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, probs: Tensor, text: List[str], log_probs_length, *args, **kwargs):
        wers = []
        lenghts = log_probs_length.tolist()
        for prob_vec, target_text, lenght in zip(probs, text, lenghts):
            prob = prob_vec[:lenght]
            pred_text = self.text_encoder.ctc_beam_search(prob)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)