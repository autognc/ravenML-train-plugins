import numpy as np
from collections import defaultdict
import json
import os
from pathlib import Path
import math

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


def calculate_statistics(all_truths, all_detections, category_index):
    confidence = defaultdict()
    accuracy = defaultdict()
    recall = defaultdict()
    precision = defaultdict()
    iou = defaultdict()
    parameters = defaultdict()

    for class_id in category_index:
        class_name = category_index[class_id]['name']
        iou[class_name] = []
        confidence[class_name] = []

        TP = 0
        TN = 0
        FN = 0
        FP = 0

        for truth, detected in zip(all_truths, all_detections):
            # true positive
            if class_name in detected and class_name in truth:
                TP += 1
                iou[class_name].append(get_iou(truth[class_name].box_norm, detected[class_name].box))

                confidence[class_name].append(detected[class_name].score)

            # false positive, we want low scores
            elif class_name in detected and class_name not in truth:
                FP += 1
                confidence[class_name].append(1 - detected[class_name].score)
            
            # false negative, do we want score = 0?? idk
            elif class_name not in detected and class_name in truth:
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

        
        accuracy[class_name] = acc
        recall[class_name] = rec
        precision[class_name] = prec
        parameters[class_name] = (TP, FP, FN, TN)

    return confidence, accuracy, recall, precision, iou, parameters


def write_stats_to_json(confidence, accuracy, recall, precision, iou, parameters, times, category_index, output_path):

    stats = defaultdict()
    stats['initialization_time'] = round(times[0],3)
    stats['inference_time_avg'] = round(sum(times[1:]) / len(times[1:]), 3)

    for class_id in category_index:
        class_name = category_index[class_id]['name']
        
        if sum(parameters[class_name]) != 0:
            num_instances = sum(parameters[class_name])
            TP, FP, FN, TN = parameters[class_name]
            avg_confidence = float(round(np.average(confidence[class_name]), 3))
            acc_stat = round(accuracy[class_name], 3)
            recall_stat = round(recall[class_name], 3)
            precision_stat = round(precision[class_name], 3)
            avg_iou = float(round(np.average(iou[class_name]), 3))

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
        
        stats[class_name] = defaultdict()
        #stats[class_name]['name'] = category_index[class_id]['name']
        stats[class_name]['class_stats'] = defaultdict()
        
        for key in class_stat:
            stats[class_name]['class_stats'][key] = class_stat[key]

    os.makedirs(Path(output_path), exist_ok=True)

    json_path = Path(output_path) / 'stats.json'
    with open(json_path, 'w') as fp:
        json.dump(stats, fp, indent=4)
