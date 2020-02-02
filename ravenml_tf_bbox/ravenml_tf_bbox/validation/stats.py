import numpy as np
import matplotlib.pyplot as plt
from object_detection.metrics.coco_evaluation import CocoDetectionEvaluator
from object_detection.core.standard_fields import InputDataFields, DetectionResultFields
from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluator


def _get_area(box):
    x = box[0] - box[2]
    y = box[1] - box[3]
    return x*y


def _get_iou(t, d):
    # xmin, ymin, xmax, ymax
    # union box
    u = (min(t['xmin'], d['xmin']), min(t['ymin'], d['ymin']), max(t['xmax'], d['xmax']), max(t['ymax'], d['ymax']))

    # intersection box
    i = (max(t['xmin'], d['xmin']), max(t['ymin'], d['ymin']), min(t['xmax'], d['xmax']), min(t['ymax'], d['ymax']))
    
    iou = _get_area(i) / _get_area(u)
    return iou


def _get_precision(cf):
    if cf['true_positive'] + cf['false_positive'] + cf['misplaced_positive'] > 0:
        return cf['true_positive'] / (cf['true_positive'] + cf['false_positive'] + cf['misplaced_positive'])
    return 0


def _get_recall(cf):
    if cf['true_positive'] + cf['false_negative'] + cf['misplaced_positive'] > 0:
        return cf['true_positive'] / (cf['true_positive'] + cf['false_negative'] + cf['misplaced_positive'])
    return 0


def _get_centroid(bbox):
    return (bbox['ymin'] + bbox['ymax']) / 2, (bbox['xmin'] + bbox['xmax']) / 2


def _get_distance(centroid_a, centroid_b):
    ay, ax = centroid_a
    by, bx = centroid_b
    return np.sqrt((ay - by) ** 2 + (ax - bx) ** 2)


def calculate_coco_statistics(all_outputs, all_bboxes, category_index):
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
            bbox = bbox_dict.get(class_name)  # bbox for this class

            # add ground truth for this class and this image to the evaluator
            if bbox:
                boxes = [[bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax']]]
            else:
                boxes = [[]]
            groundtruth_dict = {
                InputDataFields.groundtruth_boxes: np.array(boxes, dtype=np.float32),
                InputDataFields.groundtruth_classes: np.full(1, class_id, dtype=np.float32)
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
    confusion_matrix = {cls['name']: {
        'true_positive': 0,
        'false_positive': 0,
        'true_negative': 0,
        'false_negative': 0,
        'misplaced_positive': 0
    } for cls in category_index.values()}
    for output_dict, bbox_dict in zip(all_outputs, all_bboxes):
        for class_id in category_index.keys():
            class_name = category_index[class_id]['name']
            outputs = output_dict[class_name]  # list of (score, bbox) for this class
            bbox = bbox_dict.get(class_name)

            # NOTE: normally, we would filter all of the outputs by the confidence threshold, and any additional
            # detections beyond the correct one would be counted as false positives. However, since we know that there's
            # only one instance of each class, we're just going to take the top prediction and ignore the rest. This
            # will decrease the false positive rate by a little bit, but it's representative of our actual application,
            # since we will never look at more than the top prediction when we know there's only one spacecraft.
            if outputs and outputs[0][0] >= confidence_threshold:
                # the model made a detection, so check if it's correct
                if bbox:
                    if _get_iou(bbox, outputs[0][1]) >= iou_threshold:
                        confusion_matrix[class_name]['true_positive'] += 1
                    else:
                        confusion_matrix[class_name]['misplaced_positive'] += 1
                else:
                    confusion_matrix[class_name]['false_positive'] += 1
            else:
                # the model did not make a detection, so check if there was actually something there
                if bbox:
                    confusion_matrix[class_name]['false_negative'] += 1
                else:
                    confusion_matrix[class_name]['true_negative'] += 1
    return confusion_matrix


def calculate_truth_bbox_to_truth_centroid_error(all_bboxes, all_centroids, category_index):
    distances = {cls['name']: [] for cls in category_index.values()}
    for bbox_dict, centroid_dict in zip(all_bboxes, all_centroids):
        for class_id in category_index.keys():
            class_name = category_index[class_id]['name']
            bbox = bbox_dict.get(class_name)
            centroid = centroid_dict.get(class_name)
            if not (bbox and centroid):
                continue
            distances[class_name].append(_get_distance(_get_centroid(bbox), centroid))
    for k, v in distances.items():
        distances[k] = np.mean(np.array(distances[k]))
    return distances


def calculate_distance_statistics(all_outputs, all_bboxes, all_centroids, category_index, confidence_threshold):
    stats = {cls['name']: {
        'bbox_to_bbox': [],
        'bbox_to_centroid': []
    } for cls in category_index.values()}
    for output_dict, bbox_dict, centroid_dict in zip(all_outputs, all_bboxes, all_centroids):
        for class_id in category_index.keys():
            class_name = category_index[class_id]['name']
            outputs = output_dict[class_name]  # list of (score, bbox) for this class
            bbox = bbox_dict.get(class_name)
            centroid = centroid_dict.get(class_name)
            # this will only consider true positives
            if not (bbox and centroid and outputs[0][0] >= confidence_threshold):
                continue

            stats[class_name]['bbox_to_bbox'].append(
                _get_distance(_get_centroid(outputs[0][1]), _get_centroid(bbox))
            )
            stats[class_name]['bbox_to_centroid'].append(
                _get_distance(_get_centroid(outputs[0][1]), centroid)
            )

    # average everything
    for class_name, values in stats.items():
        for key, value in values.items():
            stats[class_name][key] = np.mean(np.array(value))

    return stats


def plot_pr_curve(class_name, all_outputs, all_bboxes, category_index, iou_thresholds=(0.1, 0.25, 0.5, 0.75, 0.9)):
    plt.clf()
    fig, ax = plt.subplots()
    for iou_threshold in iou_thresholds:
        precisions = []
        recalls = []
        for score in np.arange(0, 1.025, 0.025)[::-1]:
            cf = calculate_confusion_matrix(all_outputs, all_bboxes, category_index, score, iou_threshold)[class_name]
            recalls.append(_get_recall(cf))
            precisions.append(_get_precision(cf))
        ax.scatter(np.array(recalls), np.array(precisions), 10, label=str(iou_threshold))
    ax.set_title(f'{class_name} PR Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.legend(title="IOU Threshold", loc='upper left')


def plot_dr_curve(class_name, all_outputs, all_bboxes, all_centroids, category_index):
    plt.clf()
    fig, ax = plt.subplots()
    recalls = []
    distances = []
    for score in np.arange(0, 1.025, 0.025)[::-1]:
        # get recall ignoring IOU (meaning all detections when a truth bbox exists somewhere in the image
        # are counted as true positive)
        cf = calculate_confusion_matrix(all_outputs, all_bboxes, category_index, score, 0.0)[class_name]
        recalls.append(_get_recall(cf))
        # get average distances
        distances.append(
            calculate_distance_statistics(all_outputs, all_bboxes, all_centroids, category_index, score)[class_name]
        )
    distances = {k: [d[k] for d in distances] for k in distances[0].keys()}
    for k, v in distances.items():
        ax.scatter(np.array(recalls), np.array(distances[k]), 10, label=k)
    ax.set_title(f'{class_name} DR Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Distance')
    plt.legend(title="Distance Type", loc='upper right')


