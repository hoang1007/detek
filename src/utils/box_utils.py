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


# def bbox_transform(src_boxes: torch.Tensor, tar_boxes: torch.Tensor):
#     """Transform the target boxes to the offset space of the source boxes.

#     Args:
#         src_boxes: Source boxes. Shape (N, 4)
#         tar_boxes: Target boxes to be transformed. Shape (N, 4)
#         NOTE: Boxes in `(x_tl, y_tl, x_br, y_br)` format
#     Returns:
#         deltas: Shape (N, 4)
#     """

#     # Unzip the boxes
#     src_cxs, src_cys, src_ws, src_hs = xyxy2cxcywh(src_boxes).T
#     tar_cxs, tar_cys, tar_ws, tar_hs = xyxy2cxcywh(tar_boxes).T

#     tx = (tar_cxs - src_cxs) / src_ws
#     ty = (tar_cys - src_cys) / src_hs
#     tw = torch.log(tar_ws / src_ws)
#     th = torch.log(tar_hs / src_hs)

#     deltas = torch.stack((tx, ty, tw, th), dim=1)

#     return deltas
def bbox_transform(reference_boxes: torch.Tensor, proposals: torch.Tensor):
    """
    Encode a set of proposals with respect to some
    reference boxes
    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


# def bbox_inv_transform(
#     src_boxes: torch.Tensor, tar_deltas: torch.Tensor, clamp_thresh=math.log(1000 / 16)
# ):
#     """Inverse transform the target boxes to the offset space of the source boxes.

#     Args:
#         src_boxes: Source boxes. Boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
#         tar_deltas: Outputs of `bbox_transform`. Shape (N, 4)
#         clamp_thresh: Clamp the maximum value of the target width and height.

#     Returns:
#         tar_boxes: Target boxes in `(x_tl, y_tl, x_br, y_br)` format. Shape (N, 4)
#     """

#     cxs, cys, ws, hs = xyxy2cxcywh(src_boxes).T
#     tx, ty, tw, th = tar_deltas.T

#     tw = torch.clamp(tw, max=clamp_thresh)
#     th = torch.clamp(th, max=clamp_thresh)

#     tar_cxs = tx * ws + cxs
#     tar_cys = ty * hs + cys
#     tar_ws = torch.exp(tw) * ws
#     tar_hs = torch.exp(th) * hs

#     tar_boxes = torch.stack((tar_cxs, tar_cys, tar_ws, tar_hs), dim=1)
#     tar_boxes = cxcywh2xyxy(tar_boxes, inplace=True)

#     return tar_boxes


def bbox_inv_transform(
    self, boxes: torch.Tensor, rel_codes: torch.Tensor, bbox_xform_clip=math.log(1000 / 16)
):
    """From a set of original boxes and encoded relative box offsets, get the decoded boxes.

    Args:
        rel_codes (Tensor): encoded boxes
        boxes (Tensor): reference boxes.
    """

    boxes = boxes.to(rel_codes.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = self.weights
    dx = rel_codes[:, 0::4] / wx
    dy = rel_codes[:, 1::4] / wy
    dw = rel_codes[:, 2::4] / ww
    dh = rel_codes[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    # Distance from center to box's corner.
    c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

    pred_boxes1 = pred_ctr_x - c_to_c_w
    pred_boxes2 = pred_ctr_y - c_to_c_h
    pred_boxes3 = pred_ctr_x + c_to_c_w
    pred_boxes4 = pred_ctr_y + c_to_c_h
    pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(
        1
    )
    return pred_boxes


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
