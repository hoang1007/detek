from typing import Dict, Optional
import torch
from pytorch_lightning import LightningModule
from torch import optim

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from structures import BatchDataSample
from .detectors import BaseDetector


class DetectionModule(LightningModule):
    def __init__(
        self,
        detector: BaseDetector,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()
        self.detector = detector
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.save_hyperparameters(logger=False)

        self.ap_metric = MeanAveragePrecision()

    def forward(
        self,
        images,
        gt_bboxes: Optional[torch.Tensor] = None,
        gt_labels: Optional[torch.Tensor] = None,
    ):
        if self.training:
            assert gt_bboxes is not None
            assert gt_labels is not None
            return self.detector.forward_train(images, gt_bboxes, gt_labels)
        else:
            return self.detector.forward_test(images)

    def training_step(self, batch: BatchDataSample, batch_idx: int):
        batch.to(self.device)

        loss_dict = self(batch.images, batch.bboxes, batch.labels)
        assert isinstance(loss_dict, Dict), "loss_dict must be a dict"

        self.log_dict(loss_dict, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch.batch_size)

        loss = sum(loss_dict.values())
        return loss

    def validation_step(self, batch: BatchDataSample, batch_idx: int):
        batch.to(self.device)

        pred_bboxes, pred_labels, pred_scores = self(batch.images)

        preds = []
        gts = []
        for i in range(pred_bboxes.shape[0]):
            preds.append(
                {
                    "boxes": pred_bboxes[i],
                    "labels": pred_labels[i],
                    "scores": pred_scores[i],
                }
            )

            gts.append(
                {
                    "boxes": batch.bboxes[i],
                    "labels": batch.labels[i],
                }
            )

        self.ap_metric.update(preds, gts)

    def validation_epoch_end(self, outputs):
        self.log(
            "mAP", self.ap_metric.compute(), prog_bar=True, on_epoch=True, on_step=False
        )

    def test_step(self, batch: BatchDataSample, batch_idx: int):
        batch.to(self.device)

        return self(batch.images)

    def configure_optimizers(self):
        if self.hparams.optimizer is not None:  # type: ignore
            optimizer = self.hparams.optimizer(params=self.parameters())  # type: ignore
            if self.hparams.scheduler is not None:  # type: ignore
                scheduler = self.hparams.scheduler(optimizer=optimizer)  # type: ignore
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }

            return {"optimizer": optimizer}
        else:
            raise ValueError("No optimizer specified")
