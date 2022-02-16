import torch
from collections import Counter
from YOLO.metrics.iou import intersection_over_union


# MAP (Mean Average Precision) : most common metric in Deep Learning to evaluate Object Detection Models
def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint",
                           num_classes=20):
    """
    This function calculates mean average precision (mAP)
    Algorithm Steps :
        1. Get all Bounding Box predictions on our test set
            - Create a Table including columns : Image, Confidence and TP/FP
        2. Sort by descending Confidence Score
        3. Precision and Recall (Consideration of TP, FN and FP only...and not th TN)
            (Reason : TN means Object Detection Algorithm is not showing any Bounding Box for No Bounding Box in Ground
            Truth....which is obvious and so ignored TN from any Evaluation Criteria)
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
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        - true_boxes (list): Similar as pred_boxes except all the correct ones
        - iou_threshold (float): threshold where predicted bboxes is correct
        - box_format (str): "midpoint" or "corners" used to specify bboxes
        - num_classes (int): number of classes
    :return:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        print('C : ', c)
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # train_idx : is a 1st index OR Training Image Index (which will be 0 for Img_0, 1 for Img_1 and so on..)
        # Explanation of 'amount_b_boxes': find the amount of b_boxes for each training example
        # Counter here finds how many ground truth b_boxes we get for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with: amount_b_boxes = {0:3, 1:5}
        # print('ground_truths : ', ground_truths)
        amount_b_boxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # amount_b_boxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_b_boxes.items():
            amount_b_boxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_b_boxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_b_boxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                b_box for b_box in ground_truths if b_box[0] == detection[0]
            ]
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_b_boxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_b_boxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        tp_cum_sum = torch.cumsum(TP, dim=0)
        fp_cum_sum = torch.cumsum(FP, dim=0)

        # Precision on Y axis, Recall on X axis, Observe Average_Precisions concept figure in 'figures' folder
        precisions = torch.divide(tp_cum_sum, (tp_cum_sum + fp_cum_sum + epsilon))
        # Precision Starts from 1 so concatenating tensor([1]) here
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = tp_cum_sum / (total_true_b_boxes + epsilon)
        # Precision Starts from 0 so concatenating tensor([0]) here
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
    return sum(average_precisions) / len(average_precisions)