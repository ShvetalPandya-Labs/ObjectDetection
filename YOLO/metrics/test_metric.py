import torch
print(torch.__version__)
input()

import numpy as np
from YOLO.metrics.iou import intersection_over_union
from YOLO.metrics.nonmaxsuppression import non_max_suppression
from YOLO.metrics.meanavgprecision import mean_average_precision


def test_iou():
    # Test cases for 'midpoint'
    box_format = 'midpoint'
    # Test cases for 'corners'
    # box_format = 'corners'

    if box_format == 'midpoint':
        boxes_preds = torch.tensor([[0.4, 0.4, 0.3, 0.3]])
        boxes_labels = torch.tensor([[0.5, 0.5, 0.3, 0.3]])

    if box_format == 'corners':
        # boxes_preds = torch.tensor([[0.2, 0.3, 0.7, 0.8]])
        # boxes_labels = torch.tensor([[0.4, 0.1, 0.9, 0.5]])

        boxes_preds = torch.tensor([[0.3, 0.3, 0.7, 0.7]])
        boxes_labels = torch.tensor([[0.4, 0.4, 0.9, 0.9]])

        # No Intersection example
        # boxes_preds = torch.tensor([[0.1, 0.1, 0.2, 0.2]])
        # boxes_labels = torch.tensor([[0.3, 0.3, 0.4, 0.4]])

    return intersection_over_union(boxes_preds, boxes_labels, box_format)


def test_nms():
    """
    - bboxes (list): list of lists containing all bboxes with each bboxes
    specified as [class_pred, prob_score, x1, y1, x2, y2]
    - iou_threshold (float): threshold where predicted bboxes is correct
    - prob_threshold (float): threshold to remove predicted bboxes (independent of IoU)
    - box_format (str): "midpoint" or "corners" used to specify bboxes
    """

    iou_threshold = 0.5
    prob_threshold = 0.6
    bb0 = [1, 0.2, 0.5, 0.5, 0.4, 0.4]
    bb1 = [1, 0.62, 0.52, 0.52, 0.3, 0.3]
    bb2 = [1, 0.75, 0.55, 0.53, 0.4, 0.4]
    bb3 = [1, 0.89, 0.5, 0.5, 0.4, 0.4]

    bb4 = [2, 0.78, 0.85, 0.76, 0.5, 0.5]
    bb5 = [2, 0.9, 0.8, 0.8, 0.5, 0.5]
    b_boxes = [bb0, bb1, bb2, bb3, bb4, bb5]

    box_format = "midpoint"

    return non_max_suppression(b_boxes, iou_threshold, prob_threshold, box_format)


def test_map():
    """
    - pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    - train_idx : Image index which will be 0 for Img_0, 1 for Img_1 and so on..
    -

    """
    bb0 = [0, 1, 0.93, 0.82, 0.87, 0.3, 0.29]
    bb1 = [0, 1, 0.5, 0.75, 0.85, 0.25, 0.35]
    bb2 = [0, 1, 0.62, 0.52, 0.52, 0.3, 0.3]
    bb3 = [0, 1, 0.75, 0.55, 0.53, 0.4, 0.4]
    bb4 = [0, 1, 0.89, 0.51, 0.495, 0.42, 0.45]

    bb5 = [1, 2, 0.78, 0.25, 0.78, 0.4, 0.4]
    bb6 = [1, 2, 0.9, 0.31, 0.86, 0.35, 0.45]
    pred_boxes = [bb0, bb1, bb2, bb3, bb4, bb5, bb6]

    num_classes = 20
    true_boxes = [[0, 1, 1, 0.85, 0.85, 0.3, 0.3], [0, 1, 1, 0.5, 0.5, 0.4, 0.4],  [1, 2, 1, 0.2, 0.8, 0.3, 0.3]]

    iou_threshold_list = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    print('iou_threshold_list : ', iou_threshold_list)
    iou_threshold_map_dct = {}
    for iou_threshold in iou_threshold_list:
        box_format = "midpoint"
        map = mean_average_precision(pred_boxes, true_boxes, iou_threshold, box_format, num_classes)
        iou_threshold_map_dct.update({iou_threshold:map})
        print('map : ', map)
        # input()
    return iou_threshold_map_dct


if __name__ == '__main__':
    print('Enter Test function name (, nms ):\n(a) "iou" for "Intersection Over Union", '
          '\n(b)"nms" for "Non-Max Suppression",  \n(c)"map" for "Mean Average Precision" ')
    test_function_name = input()
    if test_function_name == 'iou':
        iou = test_iou()
        print('IOU : ', iou)
    elif test_function_name == 'nms':
        nms_output = test_nms()
        print(nms_output)
    elif test_function_name == 'map':
        iou_threshold_map_dct = test_map()
        print(iou_threshold_map_dct)
