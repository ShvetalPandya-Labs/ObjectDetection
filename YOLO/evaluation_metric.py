import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(b_boxes, iou_threshold, prob_threshold, box_format="corners"):
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


# MAP (Mean Average Precision) : most common metric in Deep Learning to evaluate Object Detection Models
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners",
                           num_classes=20):
    """
    This function calculates mean average precision (mAP)
    Algorithm Steps :
        1. Get all Bounding Box predictions on our test set
            - Create a Table including columns : Image, Confidence and TP/FP
        2. Sort by descending Confidence Score
        3. Precision and Recall (Consideration of TP, FN and FP only...and not th TN)
            (Reason : TN means Object Detection Algorithm is not showing any Bounding Box for No Bounding Box in Ground Truth
            ....whcih is obvious and so ignored TN from any Evaluation Criteria)
            - Precision = TP / TP + FP (of all Bounding Box predictions what fraction was actually correct !! )
            - Recall = TP / TP + FN (of all of the Ground Truth/Target Bounding Boxes what fraction did we correctly detect !! )
            Some applications prioritize Precision and some applications prioritize Recall
            - For Ex : In Self Driving Car Application -> prioritize 'Not missing any Pedestrian' -> means the application proioritize High Recall

            * calculate Precision and Recall for all Predicted Bounding Boxes
            - Update Table with Columns : Precision and Recall
            - Update Precision and Recall values for each Image's predicted Bounding Boxes
        4. Plot the Precision-Recall Graph
        5. Calculate CLASS-wise AP (Average Precision) which is the area under the Precision-Recall graph
        6. Take a mean of all 'Average Predictions' calculated for all classes
        7. Redo all computations for many IoUs : For ex : 0.5, 0.55, 0.60....0.95. Then average this to calculate final Result.
            - sometimes in paper it would be written as : mAP@0.5:0.05:0.95

    Parameters:
        - pred_boxes (list): list of lists containing all bboxes with each bboxes
        - specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        - true_boxes (list): Similar as pred_boxes except all the correct ones
        - iou_threshold (float): threshold where predicted bboxes is correct
        - box_format (str): "midpoint" or "corners" used to specify bboxes
        - num_classes (int): number of classes
    :return:
        float: mAP value across all classes given a specific IoU threshold
    """

    # pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ...]
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        print('C : ', c)
        print('true_boxes : ', true_boxes)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # image 0 has 3 b_boxes
        # image 1 has 5 b_boxes
        # amount_b_boxes = {0:3, 1:5}
        print('ground_truths : ', ground_truths)
        amount_b_boxes = Counter([gt[0] for gt in ground_truths])
        print()
        print('amount_b_boxes: ', amount_b_boxes)
        input()
        for key, val in amount_b_boxes.items():
            print('key : ', key)
            print('val : ', val)
            amount_b_boxes[key] = torch.zeros(val)


        print('---> amount_b_boxes: ', amount_b_boxes)
        input()

        # amount_boxes = {0:torch.tensor([0, 0, 0]), 1:torch.tensor([0, 0, 0, 0, 0])}

    return