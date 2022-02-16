import torch
from YOLO.metrics.iou import intersection_over_union


def non_max_suppression(b_boxes, iou_threshold, prob_threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given b_boxes
    logic of Algorithm :
        - Discard all bounding Boxes < probability Threshold
        - while b_boxes:
            - Take out the Largest Probability Box
            - Remove all other boxes with IOU >  iou_threshold
    Parameters:
        b_boxes (list): list of lists containing all b_boxes with each b_boxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted b_boxes is correct
        prob_threshold (float): threshold to remove predicted b_boxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify b_boxes
    Returns:
        list: b_boxes after performing NMS given a specific IoU threshold
    """

    assert type(b_boxes) == list

    b_boxes = [box for box in b_boxes if box[1] > prob_threshold]
    b_boxes = sorted(b_boxes, key=lambda x: x[1], reverse=True)
    b_boxes_after_nms = []

    while b_boxes:
        chosen_box = b_boxes.pop(0)

        b_boxes = [
            box
            for box in b_boxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            ) < iou_threshold
        ]

        b_boxes_after_nms.append(chosen_box)
    return b_boxes_after_nms
