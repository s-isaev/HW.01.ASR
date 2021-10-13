import logging
from typing import List
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here

    max_spectogram_len = 0
    max_text_encoded_len = 0
    for dataset_item in dataset_items:
        max_spectogram_len = max(
            max_spectogram_len, dataset_item['spectrogram'].shape[2])
        max_text_encoded_len = max(
            max_text_encoded_len, dataset_item['text_encoded'].shape[1])

    batch_spectograms = torch.zeros(
        len(dataset_items),
        dataset_items[0]['spectrogram'].shape[1],
        max_spectogram_len
    )
    batch_text_encoded = torch.zeros(
        len(dataset_items),
        max_text_encoded_len
    )
    batch_text_encoded_length = torch.zeros(
        len(dataset_items), dtype=torch.int32
    )
    batch_spectrogram_length = torch.zeros(
        len(dataset_items), dtype=torch.int32
    )
    batch_text = []

    for i, dataset_item in enumerate(dataset_items):
        batch_spectograms[i, :, :dataset_item['spectrogram'].shape[2]] = \
            dataset_item['spectrogram'].squeeze()
        batch_text_encoded[i, :dataset_item['text_encoded'].shape[1]] = \
            dataset_item['text_encoded'].squeeze()
        batch_text_encoded_length[i] = dataset_item['text_encoded'].shape[1]
        batch_spectrogram_length[i] = dataset_item['spectrogram'].shape[2]
        batch_text.append(dataset_item['text'])

    result_batch['spectrogram'] = torch.swapaxes(batch_spectograms, 1, 2)
    result_batch['text_encoded'] = batch_text_encoded
    result_batch['text_encoded_length'] = batch_text_encoded_length
    result_batch['spectrogram_length'] = batch_spectrogram_length
    result_batch['text'] = batch_text

    return result_batch
