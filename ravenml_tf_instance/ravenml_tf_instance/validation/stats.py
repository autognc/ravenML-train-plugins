import numpy as np
from collections import defaultdict
import json
import os
from pathlib import Path

def get_iou(truth_mask, detected_mask):
    truth_mask = np.array(truth_mask, dtype=bool)
    detected_mask = np.array(detected_mask, dtype=bool)

    intersection = truth_mask * detected_mask
    union = truth_mask + detected_mask

    iou = round(intersection.sum()/float(union.sum()), 3)

    return iou


def calculate_statistics(all_truths, all_detections, category_index):
    confidence = defaultdict()
    accuracy = defaultdict()
    recall = defaultdict()
    precision = defaultdict()
    iou = defaultdict()
    parameters = defaultdict()

    for class_id in category_index:
        iou[class_id] = []
        confidence[class_id] = []
        TP = 0
        TN = 0
        FN = 0
        FP = 0

        for truth, detected in zip(all_truths, all_detections):

            # true positive, we want high scores and to compute iou
            if class_id in detected and class_id in truth:
                TP += 1

                truth_mask = truth[class_id].mask
                detected_mask = detected[class_id].mask
                iou[class_id].append(get_iou(truth_mask, detected_mask))

                confidence[class_id].append(detected[class_id].score)

            # false positive, we want low scores
            elif class_id in detected and class_id not in truth:
                FP += 1
                confidence[class_id].append(100.000 - detected[class_id].score)
            
            # false negative, do we want score = 0?? idk
            elif class_id not in detected and class_id in truth:
                FN += 1

            # true negative, do we want score = 100?? idk
            else:
                TN += 1

        if TP+FP+FN+TN == 0:
            acc = 0
        else:
            acc = (TP+TN) / (TP+FP+FN+TN)
        
        if TP+FN == 0:
            rec = 0
        else:
            rec = TP / (TP+FN)

        if TP+FP == 0:
            prec = 0
        else:
            prec = TP / (TP+FP)

        
        accuracy[class_id] = acc
        recall[class_id] = rec
        precision[class_id] = prec
        parameters[class_id] = (TP, FP, FN, TN)


    return confidence, accuracy, recall, precision, iou, parameters


def write_stats_to_json(confidence, accuracy, recall, precision, iou, parameters, times, category_index, output_path):

    stats = defaultdict()
    stats['initialization_time'] = times[0]
    stats['inference_time_avg'] = round(sum(times[1:]) / len(times[1:]), 3)

    for class_id in category_index:
        
        if sum(parameters[class_id]) != 0:
            num_instances = sum(parameters[class_id])
            TP, FP, FN, TN = parameters[class_id]
            avg_confidence = round(np.average(confidence[class_id]), 3)
            acc_stat = round(accuracy[class_id], 3)
            recall_stat = round(recall[class_id], 3)
            precision_stat = round(precision[class_id], 3)
            avg_iou = round(np.average(iou[class_id]), 3)

        else:
            num_instances = 0
            TP, FP, FN, TN = 0, 0, 0, 0
            avg_confidence = 0.000
            acc_stat = 0
            recall_stat = 0.000
            precision_stat = 0.000
            avg_iou = 0.000
        
        class_stat = {"num_instances": num_instances,
                      "avg_confidence": avg_confidence,
                      "accuracy": acc_stat,
                      "recall": recall_stat,
                      "precision": precision_stat,
                      "avg_iou": avg_iou}

        class_stat["parameters"] = defaultdict()
        class_stat["parameters"]['true_positive'] = TP
        class_stat["parameters"]['false_positive'] = FP
        class_stat["parameters"]['false_negative'] = FN
        class_stat["parameters"]['true_negative'] = TN
        
        stats[class_id] = defaultdict()
        stats[class_id]['name'] = category_index[class_id]['name']
        stats[class_id]['class_stats'] = defaultdict()
        
        for key in class_stat:
            stats[class_id]['class_stats'][key] = class_stat[key]

    os.makedirs(Path(output_path), exist_ok=True)

    json_path = Path(output_path) / 'stats.json'
    with open(json_path, 'w') as fp:
        json.dump(stats, fp, indent=4)
