import random
from random import shuffle

import PIL
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_asr.base import BaseTrainer
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics_train,
            metrics_valid,
            optimizer,
            config,
            device,
            data_loader,
            text_encoder,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            log_step=None
    ):
        super().__init__(model, criterion, metrics_train, metrics_valid, optimizer, config, device)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 10 if log_step is None else log_step

        self.train_metrics_tracker = MetricTracker(
            "loss", "grad norm",
            *[m.name for m in self.metrics_train], writer=self.writer
        )
        self.train_metrics_log_step_tracker = MetricTracker(
            "loss", "grad norm",
            *[m.name for m in self.metrics_train], writer=self.writer
        )
        self.valid_metrics_tracker = MetricTracker(
            "loss",
            *[m.name for m in self.metrics_valid], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in [
            "spectrogram",
            "text_encoded"
        ]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_iteration(self, batch: dict, epoch: int, batch_num: int):
        if batch_num  % self.log_step == 0:
            self.train_metrics_log_step_tracker.reset()

        batch = self.move_batch_to_device(batch, self.device)
        self.optimizer.zero_grad()

        batch["logits"] = self.model(**batch)
        batch["probs"] = F.softmax(batch["logits"], dim=-1)
        batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        batch["log_probs_length"] = self.model.transform_input_lengths(
            batch["spectrogram_length"]
        )

        loss = self.criterion(**batch)
        loss.backward()
        self._clip_grad_norm()
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.train_metrics_tracker.update("loss", loss.item())
        self.train_metrics_log_step_tracker.update("loss", loss.item())
        for met in self.metrics_train:
            met_val = met(**batch)
            self.train_metrics_tracker.update(met.name, met_val)
            self.train_metrics_log_step_tracker.update(met.name, met_val)
        grad_norm = self.get_grad_norm()
        self.train_metrics_tracker.update("grad norm", grad_norm)
        self.train_metrics_log_step_tracker.update("grad norm", grad_norm)

        # Каждый раз делаем set step, чтобы работало steps per second
        self.writer.set_step((epoch - 1) * self.len_epoch + batch_num, mode="train")

        if (batch_num + 1) % self.log_step == 0:
            self.logger.debug(
                "Train Epoch: {} {} Loss: {:.6f}".format(
                    epoch, self._progress(batch_num), loss.item()
                )
            )
            self.writer.add_scalar(
                "learning rate", self.lr_scheduler.get_last_lr()[0]
            )
            self._log_predictions(part="train", **batch)
            self._log_spectrogram(batch["spectrogram"])
            self._log_scalars(self.train_metrics_log_step_tracker)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics_tracker.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            try:
                self._train_iteration(batch, epoch, batch_idx)
            except RuntimeError as e:
                if 'out of memory' in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx >= self.len_epoch:
                break

        log = self.train_metrics_tracker.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics_tracker.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader), desc="validation",
                    total=len(self.valid_data_loader)
            ):
                batch = self.move_batch_to_device(batch, self.device)
                batch["logits"] = self.model(**batch)
                batch["probs"] = F.softmax(batch["logits"], dim=-1)
                batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
                batch["log_probs_length"] = self.model.transform_input_lengths(
                    batch["spectrogram_length"]
                )
                
                loss = self.criterion(**batch)

                self.valid_metrics_tracker.update("loss", loss.item(), n=len(batch["text"]))
                for met in self.metrics_valid:
                    self.valid_metrics_tracker.update(met.name, met(**batch))
            self.writer.set_step(epoch * self.len_epoch, mode="valid")
            self._log_scalars(self.valid_metrics_tracker)
            self._log_predictions(part="val", **batch)
            self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics_tracker.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            probs,
            log_probs,
            log_probs_length,
            examples_to_log=5,
            *args,
            **kwargs,
    ):
        # TODO: implement logging of beam search results
        if self.writer is None:
            return

        predictions_log_probs = log_probs.cpu().argmax(-1).tolist()
        for i, log_len in enumerate(log_probs_length.tolist()):
            predictions_log_probs[i] = predictions_log_probs[i][:log_len]

        c_probs = probs.cpu()
        predictions_probs = []
        for i, log_len in enumerate(log_probs_length.tolist()):
            predictions_probs.append(c_probs[i][:log_len])

        ctc_pred_text = [self.text_encoder.ctc_decode(p) for p in predictions_log_probs]
        raw_pred_text = [self.text_encoder.decode(p) for p in predictions_log_probs]
        beam_pred_text = [self.text_encoder.ctc_beam_search(p) for p in predictions_probs]

        tuples = list(zip(text, raw_pred_text, ctc_pred_text, beam_pred_text))
        shuffle(tuples)
        to_log_pred_ctc = []
        to_log_pred_raw = []
        to_log_pred_beam = []
        for target, raw_pred, ctc_pred, beam_pred in tuples[:examples_to_log]:
            wer = calc_wer(target, beam_pred) * 100
            cer = calc_cer(target, beam_pred) * 100
            to_log_pred_beam.append(f"true: '{target}' | pred: '{beam_pred}' " f"| wer: {wer:.2f} | cer: {cer:.2f}")

            wer = calc_wer(target, ctc_pred) * 100
            cer = calc_cer(target, ctc_pred) * 100
            to_log_pred_ctc.append(f"true: '{target}' | pred: '{ctc_pred}' " f"| wer: {wer:.2f} | cer: {cer:.2f}")

            to_log_pred_raw.append(f"true: '{target}' | pred: '{raw_pred}'")

        self.writer.add_text(f"predictions_beam", '\n\n'.join(to_log_pred_beam))
        self.writer.add_text(f"predictions_ctc", '\n\n'.join(to_log_pred_ctc))
        self.writer.add_text(f"predictions_raw", '\n\n'.join(to_log_pred_raw))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch)
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.cpu().log()))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
