import torch

from structures import ImageInfo
from utils.box_utils import bbox_iou, bbox_transform
from utils.functional import random_choice


class AnchorTargetGenerator:
    def __init__(
        self,
        batch_size: int = 128,
        allowed_border: float = 0,
        clobber_positives: bool = False,
        positive_overlap: float = 0.7,
        negative_overlap: float = 0.3,
        fg_fraction: float = 0.5,
    ):
        """Gán nhãn cho từng anchor.

        Args:
            anchors: Shape (N, 4)
            gt_boxes: Hộp chân trị của các đối tượng trong ảnh. Shape (K, 4)
            im_info: Thông tin về ảnh ở định dạng (H, W, scale)
        Returns:
            bbox_targets: Transform của anchors và gt_boxes. Shape (N, 4)
            labels: Nhãn nhị phân của từng hộp. Shape (N,)
        """
        self._allowed_border = allowed_border
        self.batch_size = batch_size
        self.clobber_positives = clobber_positives
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.fg_fraction = fg_fraction

    def __call__(self, anchors: torch.Tensor, gt_boxes: torch.Tensor, metadata: ImageInfo):
        A = anchors.size(0)

        anchors, keep_ids = self._get_inside_anchors(anchors, metadata.height, metadata.width)

        bbox_targets, labels = self._mklabels(anchors, gt_boxes)

        bbox_targets = self._unmap(bbox_targets, A, keep_ids, fill=0)
        labels = self._unmap(labels, A, keep_ids, fill=-1)

        return bbox_targets, labels

    def _get_inside_anchors(self, anchors: torch.Tensor, height: int, width: int):
        inside_ids = torch.where(
            (anchors[:, 0] >= -self._allowed_border)
            & (anchors[:, 1] >= -self._allowed_border)
            & (anchors[:, 2] < width + self._allowed_border)
            & (anchors[:, 3] < height + self._allowed_border)
        )[0]

        return anchors[inside_ids], inside_ids

    def _mklabels(self, anchors: torch.Tensor, gt_boxes: torch.Tensor):
        A = anchors.size(0)
        G = gt_boxes.size(0)

        assert A > 0, "Num of anchors must be greater than 0"
        assert G > 0, "Num of ground-truth boxes must be greater than 0"

        labels = torch.empty(A, dtype=torch.long, device=anchors.device).fill_(-1)

        ious = bbox_iou(anchors, gt_boxes)  # shape (A, G)

        # lấy gt_boxes có IoU lớn nhất so với mỗi anchor
        max_ious, argmax_ious = torch.max(ious, dim=1)

        # lấy anchors có IoU lớn nhất so với mỗi gt_box
        gt_max_ious, _ = torch.max(ious, dim=0)
        gt_argmax_ious = torch.where(ious == gt_max_ious)[0]

        if not self.clobber_positives:
            labels[max_ious < self.negative_overlap] = 0

        labels[gt_argmax_ious] = 1
        labels[max_ious >= self.positive_overlap] = 1

        if self.clobber_positives:
            labels[max_ious < self.negative_overlap] = 0

        # Phân chia các nhãn: potivies, negatives, non-labels theo batch_size
        num_fg = round(self.batch_size * self.fg_fraction)
        fg_ids = torch.where(labels == 1)[0]

        if fg_ids.size(0) > num_fg:
            # Lược bỏ nếu số nhãn foreground quá nhiều
            disable_ids = fg_ids[random_choice(fg_ids, fg_ids.size(0) - num_fg, replacement=False)]
            labels[disable_ids] = -1
        else:
            # Cập nhật num_fg nếu số nhãn foreground quá ít
            num_fg = fg_ids.size(0)

        assert num_fg == (labels == 1).sum()

        num_bg = self.batch_size - num_fg
        bg_ids = torch.where(labels == 0)[0]

        if bg_ids.size(0) > num_bg:
            disable_ids = bg_ids[random_choice(bg_ids, bg_ids.size(0) - num_bg, replacement=False)]
            labels[disable_ids] = -1

        bbox_targets = bbox_transform(anchors, gt_boxes[argmax_ious])

        return bbox_targets, labels

    def _unmap(self, data: torch.Tensor, count: int, ids: torch.Tensor, fill: float = 0):
        if len(data.shape) == 1:
            ret = torch.empty((count,)).type_as(data).fill_(fill)
            ret[ids] = data
        else:
            ret = torch.empty((count,) + data.shape[1:]).type_as(data).fill_(fill)
            ret[ids, :] = data

        return ret
