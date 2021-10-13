from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, text: List[str], log_probs_length, *args, **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).tolist()
        lenghts = log_probs_length.tolist()
        for log_prob_vec, target_text, lenght in zip(predictions, text, lenghts):
            log_prob = log_prob_vec[:lenght]
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob)
            else:
                pred_text = self.text_encoder.decode(log_prob)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
