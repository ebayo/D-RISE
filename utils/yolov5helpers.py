import numpy as np
import cv2

import torch
from yolov5.utils.augmentations import letterbox

from .conversions import xywh2xyxy


def preprocess_image(image, imgsz):
    im = letterbox(image, imgsz)[0]
    # Convert
    # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = im[:, :, ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)
    # im = np.array(im)
    # Convert RGB to BGR
    # im = im[:, :, ::-1].copy()
    im = torch.from_numpy(im).to('cpu')
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im


def drise_nms(predictions: torch.tensor, iou_th=0.5, conf_th=0.25, soft=True, gaussian=False, sigma=0.5):
    """ Apply non-maxima supression to predictions on ONE image returning the objectness score and the class confidence vectors
    Based on: https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
    Class agnostic

    bbox coordinates (in and out): x1, y1, x2, y2 where x1 < x2 and y1 < y2 (top-right and bottom-left corners)
    Input:
        Tensor([num_pred_i (4 + 1 + num_classes)])
        0 <= iou_th <= 1: iou threshold
        0 <= conf_th <= 1: confidence threshold on obj_conf * cls_conf (YOLOv5 implementation)
        soft: whether using soft-nms or not (https://arxiv.org/pdf/1704.04503.pdf)
        gaussian: use gaussian weighting function instead of linear in soft-nms, ignored if using traditional nms
    Output:
        Tensor([num_pred_f (4 + 1 + num_classes)])
        num_pred_i >= num_pred_f
    """

    # Prepare predictions for nms
    scores = torch.amax(predictions[:, 5:] * predictions[:, 4, None], dim=1)
    # xc = torch.amax(predictions[:, 5:] * predictions[:, 4, None], dim=1) > conf_th
    xc = scores > conf_th
    xi = np.argwhere(np.asarray(xc))

    x1s = torch.squeeze(predictions[xi, 0])
    y1s = torch.squeeze(predictions[xi, 1])
    x2s = torch.squeeze(predictions[xi, 2])
    y2s = torch.squeeze(predictions[xi, 3])
    # boxes = torch.squeeze(predictions[xi, :4])
    objness = torch.squeeze(predictions[xi, 4])
    clsconf = torch.squeeze(predictions[xi, 5:])
    areas = (x2s - x1s) * (y2s - y1s)
    # print(areas.shape)
    # print(objness.shape)
    # print(clsconf.shape)
    scores = torch.squeeze(scores[xi[:, 0]])

    order = torch.argsort(scores)
    keep = []

    # nms algorithm
    # while len(order) > 0 and scores[order[-1]] > conf_th:
    # Second condition may not be needed, already removed smalles conf at last step of last loop step
    while len(order) > 0:
        idx = order[-1]
        box = torch.Tensor([x1s[idx], y1s[idx], x2s[idx], y2s[idx]])
        # box = torch.cat((x1s[idx].view(1), y1s[idx].view(1), x2s[idx].view(1), y2s[idx].view(1)))
        current = torch.cat((box, objness[idx].view(1), clsconf[idx]))
        keep.append(current)

        # Remove current box from potential boxes
        order = order[:-1]
        # scores[idx] = 0

        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1s, dim=0, index=order)
        xx2 = torch.index_select(x2s, dim=0, index=order)
        yy1 = torch.index_select(y1s, dim=0, index=order)
        yy2 = torch.index_select(y2s, dim=0, index=order)

        # print('\n')
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1s[idx])
        yy1 = torch.max(yy1, y1s[idx])
        xx2 = torch.min(xx2, x2s[idx])
        yy2 = torch.min(yy2, y2s[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        inter = w * h
        rem_areas = torch.index_select(areas, dim=0, index=order)
        union = (rem_areas - inter) + areas[idx]
        IoU = inter / union

        # Implementation of soft-nms
        if soft:
            if gaussian:
                weights = torch.exp(torch.pow(IoU, 1) / -sigma)
            else:   # Lineal
                weights = torch.Tensor([1 if iou < iou_th else 1-iou for iou in IoU])
        else:   # Classical
            weights = (IoU < iou_th).long()

        # Update scores (conf)
        ss = torch.index_select(scores, dim=0, index=order)
        ss *= weights
        for o, s in zip(order, ss):
            scores[o] = s
        # Remove boxes with small confidence
        mask = ss > conf_th
        order = order[mask]
        continue
    return torch.stack(keep)


def complex2simple(det):
    cls = torch.argmax(det[:, 5:], dim=1)
    conf = torch.tensor([det[i, c + 5] for i, c in enumerate(cls)])
    return torch.cat((det[:, :4], conf[:, None], cls[:, None]), dim=1)


def postprocess_detections(det, osize, lsize, simple=True):
    det[:, :4] = xywh2xyxy(det[:, :4])
    det = drise_nms(det, soft=True, gaussian=True)
    box = scale_coords(lsize, det[:, :4], osize)
    if simple:
        cls = torch.argmax(det[:, 5:], dim=1)
        conf = torch.tensor([det[i, c + 5] for i, c in enumerate(cls)])
        return torch.cat((box, conf[:, None], cls[:, None]), dim=1)
    else:
        return torch.cat((box, det[:, 4:]), dim=1)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


class Annotator:
    # YOLOv5 Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        self.colors = Colors()

    def box_label(self, box, label='', cls=0, txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        color = self.colors(cls, True)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(self.lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    def result(self):
        # Return annotated image as array
        return np.asarray(self.im)


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))