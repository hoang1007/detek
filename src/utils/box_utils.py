import math

import torch


def xyxy2cxcywh(boxes: torch.Tensor, inplace: bool = False):
    """Transform boxes from `(x_tl, y_tl, x_br, y_br)` format to `(x_ctr, y_ctr, w, h)` format.

    Args:
        boxes (Tensor): boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
        inplace (bool): If True, the boxes will be modified in place.

    Returns:
        boxes (Tensor): boxes in `(x_ctr, y_ctr, w, h)` format. Shape (N, 4)
    """
    ws = boxes[:, 2] - boxes[:, 0]  # x_br - x_tl
    hs = boxes[:, 3] - boxes[:, 1]  # y_br - y_tl
    cxs = boxes[:, 0] + 0.5 * ws  # x_tl + w / 2
    cys = boxes[:, 1] + 0.5 * hs  # y_tl + h / 2

    if inplace:
        boxes[:, 0] = cxs
        boxes[:, 1] = cys
        boxes[:, 2] = ws
        boxes[:, 3] = hs
        return boxes
    else:
        return torch.stack((cxs, cys, ws, hs), dim=1)


def cxcywh2xyxy(boxes: torch.Tensor, inplace: bool = False):
    """Transform boxes from `(x_ctr, y_ctr, w, h)` format to `(x_tl, y_tl, x_br, y_br)` format.

    Args:
        boxes (Tensor): boxes in `(x_ctr, y_ctr, w, h)` format. Shape (N, 4)
        inplace (bool): If True, the boxes will be modified in place.

    Returns:
        boxes (Tensor): boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
    """
    half_w = 0.5 * boxes[:, 2]
    half_h = 0.5 * boxes[:, 3]

    x_tl = boxes[:, 0] - half_w
    y_tl = boxes[:, 1] - half_h
    x_br = boxes[:, 0] + half_w
    y_br = boxes[:, 1] + half_h

    if inplace:
        boxes[:, 0] = x_tl
        boxes[:, 1] = y_tl
        boxes[:, 2] = x_br
        boxes[:, 3] = y_br
        return boxes
    else:
        return torch.stack((x_tl, y_tl, x_br, y_br), dim=1)


def bbox_transform(src_boxes: torch.Tensor, tar_boxes: torch.Tensor):
    """Transform the target boxes to the offset space of the source boxes.

    Args:
        src_boxes: Source boxes. Shape (N, 4)
        tar_boxes: Target boxes to be transformed. Shape (N, 4)
        NOTE: Boxes in `(x_tl, y_tl, x_br, y_br)` format
    Returns:
        deltas: Shape (N, 4)
    """

    # Unzip the boxes
    src_cxs, src_cys, src_ws, src_hs = xyxy2cxcywh(src_boxes).T
    tar_cxs, tar_cys, tar_ws, tar_hs = xyxy2cxcywh(tar_boxes).T

    assert src_ws.min() > 0 and src_hs.min() > 0, "All the source boxes should have positive width and height."
    assert tar_ws.min() > 0 and tar_hs.min() > 0, "All the target boxes should have positive width and height."

    tx = (tar_cxs - src_cxs) / src_ws
    ty = (tar_cys - src_cys) / src_hs
    tw = torch.log(tar_ws / src_ws)
    th = torch.log(tar_hs / src_hs)

    deltas = torch.stack((tx, ty, tw, th), dim=1)

    return deltas


def bbox_inv_transform(
    src_boxes: torch.Tensor, tar_deltas: torch.Tensor, clamp_thresh=math.log(1000 / 16)
):
    """Inverse transform the target boxes to the offset space of the source boxes.

    Args:
        src_boxes: Source boxes. Boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
        tar_deltas: Outputs of `bbox_transform`. Shape (N, 4)
        clamp_thresh: Clamp the maximum value of the target width and height to avoid large value in `torch.exp`.

    Returns:
        tar_boxes: Target boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
    """

    cxs, cys, ws, hs = xyxy2cxcywh(src_boxes).T
    tx, ty, tw, th = tar_deltas.T

    tw = torch.clamp(tw, max=clamp_thresh)
    th = torch.clamp(th, max=clamp_thresh)

    tar_cxs = tx * ws + cxs
    tar_cys = ty * hs + cys
    tar_ws = torch.exp(tw) * ws
    tar_hs = torch.exp(th) * hs

    tar_boxes = torch.stack((tar_cxs, tar_cys, tar_ws, tar_hs), dim=1)
    tar_boxes = cxcywh2xyxy(tar_boxes, inplace=True)

    return tar_boxes


def compute_box_areas(boxes: torch.Tensor):
    """Compute the area of boxes.

    Args:
        boxes: Boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)

    Returns:
        areas: Area of the boxes. Shape (N,)
    """
    size = boxes[:, 2:] - boxes[:, :2]
    areas = size.prod(dim=1)

    return areas


def bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Compute the Intersection over Union of two set of boxes.

    Parameters
        boxes: Boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
        query_boxes: Boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (K, 4)

    Returns
        ious: ious between boxes and query_boxes. Shape (N, K)
    """

    area1 = compute_box_areas(boxes1)
    area2 = compute_box_areas(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # shape (N, K, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # shape (N, K, 2)
    wh = (rb - lt).clamp(min=0)
    intersection = wh.prod(dim=2)  # shape (N, K)

    union = area1[:, None] + area2[None, :] - intersection
    ious = intersection / union

    assert (ious >= 0).all() and (ious <= 1).all()
    return ious


def clip_boxes(boxes: torch.Tensor, height: float, width: float):
    """Clip the boxes to the image size.

    Args:
        boxes: Boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
    """

    boxes = torch.stack(
        (
            boxes[:, 0].clamp(0, width),
            boxes[:, 1].clamp(0, height),
            boxes[:, 2].clamp(0, width),
            boxes[:, 3].clamp(0, height),
        ),
        dim=1,
    )

    return boxes
