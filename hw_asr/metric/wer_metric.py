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

    def __call__(self, probs: Tensor, log_probs: Tensor, text: List[str], log_probs_length, do_beam_search, *args, **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).tolist()
        lenghts = log_probs_length.tolist()
        for prob_vec, pred_vec, target_text, lenght in zip(probs, predictions, text, lenghts):
            if do_beam_search and hasattr(self.text_encoder, "ctc_beam_search"):
                prob = prob_vec[:lenght]
                pred_text = self.text_encoder.ctc_beam_search(prob)
            elif hasattr(self.text_encoder, "ctc_decode"):
                pred = pred_vec[:lenght]
                pred_text = self.text_encoder.ctc_decode(pred)
            else:
                pred = pred_vec[:lenght]
                pred_text = self.text_encoder.decode(pred)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
