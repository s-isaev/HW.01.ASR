from typing import List, Tuple

import torch

from hw_asr.text_encoder.char_text_encoder import CharTextEncoder
from fast_ctc_decode import beam_search


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str]):
        super().__init__(alphabet)
        self.ind2char = {
            0: self.EMPTY_TOK
        }
        self.alphabet = ['^'] + alphabet
        for text in alphabet:
            self.ind2char[max(self.ind2char.keys()) + 1] = text
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        inds_no_empty = []
        last = 0
        for ind in inds:
            if ind != 0 and ind != last:
                inds_no_empty.append(ind)
            last = ind

        res = ''
        for ind in inds_no_empty:
            res += self.ind2char[ind]
        
        return res

    def ctc_beam_search(self, probs: torch.tensor, beam_size: int = 100) -> List[Tuple[str, float]]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        
        # TODO: your code here
        return beam_search(probs.detach().cpu().numpy(), self.alphabet, beam_size=beam_size)[0]

        hypos = [('^', 1.0)]
        for el in probs:
            # creatinting new hypos
            hypo_dict = dict()
            for i, prob in enumerate(el.tolist()):
                c = self.ind2char[i]
                for hypo in hypos:
                    oldc = hypo[0]
                    oldprob = hypo[1]

                    if oldc[-1] == '^':
                        newc = oldc[:-1] + c
                    else:
                        newc = (oldc+c) if (oldc[-1]!=c) else oldc
                    newprob = oldprob*prob

                    if newc not in hypo_dict:
                        hypo_dict[newc] = newprob
                    else:
                        hypo_dict[newc] += newprob

            hypos = hypo_dict.items()
            hypos = sorted(hypos, key=lambda x: x[1], reverse=True)
            hypos = hypos[:beam_size]

            # nolmalise probs
            sumprobs = 0
            for hypo in hypos:
                sumprobs += hypo[1]
            for i in range(len(hypos)):
                hypos[i] = (hypos[i][0], hypos[i][1]/sumprobs)

        # deteting last ^
        hypo_dict = dict()
        for hypo in hypos:
            oldc = hypo[0]
            prob = hypo[1]
            newc = oldc if (oldc[-1] != '^') else oldc[:-1]

            if newc not in hypo_dict:
                hypo_dict[newc] = prob
            else:
                hypo_dict[newc] += prob
        hypos = hypo_dict.items()

        return sorted(hypos, key=lambda x: x[1], reverse=True)
