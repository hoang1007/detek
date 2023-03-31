from typing import Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from torch import optim
from torchmetrics import Accuracy
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.structures import BatchDataSample, DetResult

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
        self.accuracy = Accuracy(task="multiclass", num_classes=self.detector.num_classes)
        self.detector.init_weights()

    def forward(
        self,
        images,
        gt_bboxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
    ):
        images = self.detector.img_normalize(images)
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

        for loss_name in loss_dict.keys():
            self.log(
                f"train/{loss_name}",
                loss_dict[loss_name],
                on_epoch=True,
                on_step=False,
                batch_size=len(batch),
            )

        loss = sum(loss_dict.values())
        return loss

    def validation_step(self, batch: BatchDataSample, batch_idx: int):
        batch.to(self.device)

        det_results = self(batch.images)

        preds = []
        gts = []
        pred_labels = []
        gt_labels = []
        for det_result in det_results:
            assert isinstance(det_result, DetResult)
            preds.append(
                {
                    "boxes": det_result.bboxes,
                    "labels": det_result.labels,
                    "scores": det_result.scores,
                }
            )
            pred_labels.append(det_result.labels)

        for i in range(len(batch)):
            gts.append(
                {
                    "boxes": batch.bboxes[i],
                    "labels": batch.labels[i],
                }
            )
            gt_labels.append(batch.labels[i])

        self.ap_metric.update(preds, gts)
        self.accuracy.update(pred_labels, gt_labels)

    def validation_epoch_end(self, outputs):
        ap_metrics = self.ap_metric.compute()
        for metric_name in ap_metrics.keys():
            self.log(
                f"val/{metric_name}",
                ap_metrics[metric_name],
                prog_bar=True,
                on_epoch=True,
                on_step=False,
            )

        acc = self.accuracy.compute()
        self.log(
            "val/acc",
            acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def predict_step(self, batch: BatchDataSample):
        batch.to(self.device)

        det_results = self(batch.images)
        for image, det_result in zip(batch.images, det_results):
            assert isinstance(det_result, DetResult)
            det_result.set_image(image)
            if self.detector.CLASSES is not None:
                det_result.set_classes(self.detector.CLASSES)
        return det_results

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
