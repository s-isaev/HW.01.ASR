import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from hw_asr.datasets.utils import get_dataloaders
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
import hw_asr.model as module_model
import hw_asr.loss as module_loss
import hw_asr.metric as module_metric
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.parse_config import ConfigParser

import torch.nn.functional as F
import os
from hw_asr.metric.utils import calc_cer, calc_wer

DEFAULT_TEST_CONFIG_PATH = ROOT_PATH / "default_test_config.json"
DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # text_encoder
    text_encoder = CTCCharTextEncoder.get_simple_alphabet()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config["loss"])
    # metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    results = []
    cer_beam = []
    wer_beam = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)

            batch["logits"] = model(**batch)
            batch["probs"] = F.softmax(batch["logits"], dim=-1)
            batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )

            probs = batch["probs"]
            log_probs = batch["log_probs"]
            log_probs_length = batch["log_probs_length"]
            text = batch["text"]

            predictions_log_probs = log_probs.cpu().argmax(-1).tolist()
            for i, log_len in enumerate(log_probs_length.tolist()):
                predictions_log_probs[i] = predictions_log_probs[i][:log_len]

            c_probs = probs.cpu()
            predictions_probs = []
            for i, log_len in enumerate(log_probs_length.tolist()):
                predictions_probs.append(c_probs[i][:log_len])

            ctc_pred_text = [text_encoder.ctc_decode(p) for p in predictions_log_probs]
            raw_pred_text = [text_encoder.decode(p) for p in predictions_log_probs]
            beam_pred_text = [text_encoder.ctc_beam_search(p) for p in predictions_probs]

            for i in range(len(text)):
                results.append({
                    "ground_trurh": text[i],
                    "pred_text_raw": raw_pred_text[i],
                    "pred_text_argmax": ctc_pred_text[i],
                    "pred_text_beam_search": beam_pred_text[i]
                })
                cer_beam.append(calc_cer(text[i], beam_pred_text[i]))
                wer_beam.append(calc_wer(text[i], beam_pred_text[i]))
    print("CER:", sum(cer_beam) / len(cer_beam))
    print("WER:", sum(wer_beam) / len(wer_beam))
    with Path(out_file).open('w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=str('hw_asr/configs/test_config.json'),
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default="model_best.pth",
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default='output.json',
        type=str,
        help="File to write results (.json)",
    )

    os.system("gdown https://drive.google.com/uc?id=1CDDP8CapgqbKv5k583KRHlGswC6fyVyY")
    os.system("gdown https://drive.google.com/uc?id=1CdTBN2JkwaewCJ5NHebfu7uuGutJ1Tjr")

    config = ConfigParser.from_args(args)

    args = args.parse_args()
    config.config["data"] = {
        "test": {
            "batch_size": 40,
            "num_workers": 5,
            "datasets": [
                {
                    "type": "LibrispeechDataset",
                    "args": {
                        "part": "test-clean"
                    }
                }
            ]
        }
    }
    main(config, args.output)
