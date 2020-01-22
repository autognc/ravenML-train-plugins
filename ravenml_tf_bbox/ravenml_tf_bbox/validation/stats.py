import numpy as np
import matplotlib.pyplot as plt
from object_detection.metrics.coco_evaluation import CocoDetectionEvaluator
from object_detection.core.standard_fields import InputDataFields, DetectionResultFields
from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluator


def get_area(box):
    x = box[0] - box[2]
    y = box[1] - box[3]
    return x*y


def get_iou(t, d):
    # xmin, ymin, xmax, ymax
    # union box
    u = (min(t['xmin'], d['xmin']), min(t['ymin'],d['ymin']), max(t['xmax'], d['xmax']), max(t['ymax'], d['ymax']))

    # intersection box
    i = (max(t['xmin'], d['xmin']), max(t['ymin'],d['ymin']), min(t['xmax'], d['xmax']), min(t['ymax'], d['ymax']))
    
    iou = get_area(i) / get_area(u)
    return iou


def calculate_coco_statistics(all_outputs, all_bboxes, category_index, iou_threshold=0.5):
    # create coco evaluator
    coco_evaluator = CocoDetectionEvaluator(list(category_index.values()))
    # OD evaluator (older than coco, I left it here in case we need it later)
    """
    od_evaluator = ObjectDetectionEvaluator(
        list(category_index.values()),
        matching_iou_threshold=iou_threshold,
        evaluate_corlocs=True,
        evaluate_precision_recall=True
    )
    """
    for i, (output_dict, bbox_dict) in enumerate(zip(all_outputs, all_bboxes)):
        for class_id in category_index.keys():
            class_name = category_index[class_id]['name']
            outputs = output_dict[class_name]  # list of (score, bbox) for this class
            bboxes = bbox_dict[class_name]  # list of bboxes for this class

            # add ground truth for this class and this image to the evaluator
            boxes = [[box['ymin'], box['xmin'], box['ymax'], box['xmax']] for box in bboxes]
            groundtruth_dict = {
                InputDataFields.groundtruth_boxes: np.array(boxes, dtype=np.float32),
                InputDataFields.groundtruth_classes: np.full(len(boxes), class_id, dtype=np.float32)
            }
            coco_evaluator.add_single_ground_truth_image_info(i, groundtruth_dict)
            #  od_evaluator.add_single_ground_truth_image_info(i, groundtruth_dict)

            # add detections for this class and this image to the evaluator
            scores, boxes = zip(*outputs)
            boxes = [[box['ymin'], box['xmin'], box['ymax'], box['xmax']] for box in boxes]
            detections_dict = {
                DetectionResultFields.detection_boxes: np.array(boxes, dtype=np.float32),
                DetectionResultFields.detection_scores: np.array(scores, dtype=np.float32),
                DetectionResultFields.detection_classes: np.full(len(boxes), class_id, dtype=np.float32)
            }
            coco_evaluator.add_single_detected_image_info(i, detections_dict)
            #  od_evaluator.add_single_detected_image_info(i, detections_dict)

    return coco_evaluator.evaluate()


def calculate_confusion_matrix(all_outputs, all_bboxes, category_index, confidence_threshold, iou_threshold):
    confusion_matrix = {
        'true_positive': 0,
        'false_positive': 0,
        'true_negative': 0,
        'false_negative': 0
    }
    for output_dict, bbox_dict in zip(all_outputs, all_bboxes):
        for class_id in category_index.keys():
            class_name = category_index[class_id]['name']
            outputs = output_dict[class_name]  # list of (score, bbox) for this class
            bboxes = bbox_dict[class_name]  # list of bboxes for this class

            # NOTE: normally, we would filter all of the outputs by the confidence threshold, and any additional
            # detections beyond the correct one would be counted as false positives. However, since we know that there's
            # only one instance of each class, we're just going to take the top prediction and ignore the rest. This
            # will decrease the false positive rate by a little bit, but it's representative of our actual application,
            # since we will never look at more than the top prediction when we know there's only one spacecraft
            if outputs and outputs[0][0] >= confidence_threshold:
                # the model made a detection, so check if it's correct
                if bboxes and get_iou(bboxes[0], outputs[0][1]) >= iou_threshold:
                    confusion_matrix['true_positive'] += 1
                else:
                    confusion_matrix['false_positive'] += 1
            else:
                # the model did not make a detection, so check if there was actually something there
                if bboxes:
                    confusion_matrix['false_negative'] += 1
                else:
                    confusion_matrix['true_negative'] += 1
    return confusion_matrix


def plot_pr_curve(all_outputs, all_bboxes, category_index, iou_thresholds=(0.1, 0.25, 0.5, 0.75, 0.9)):
    plt.clf()
    fig, ax = plt.subplots()
    for iou_threshold in iou_thresholds:
        precisions = []
        recalls = []
        for score in np.arange(0, 1.025, 0.025)[::-1]:
            cf = calculate_confusion_matrix(all_outputs, all_bboxes, category_index, score, iou_threshold)
            if cf['true_positive'] + cf['false_negative'] > 0:
                recalls.append(cf['true_positive'] / (cf['true_positive'] + cf['false_negative']))
            else:
                recalls.append(0)
            if cf['true_positive'] + cf['false_positive'] > 0:
                precisions.append(cf['true_positive'] / (cf['true_positive'] + cf['false_positive']))
            else:
                precisions.append(0)
        ax.scatter(np.array(recalls), np.array(precisions), 10, label=str(iou_threshold))
    ax.set_title('PR Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.legend(title="IOU Threshold", loc='upper left')
