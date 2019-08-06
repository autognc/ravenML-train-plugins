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


def calculate_statistics(all_truths, all_detections):
    confidence = {}
    recall = {}
    precision = {}
    iou = {}

    for truth, detected in zip(all_truths, all_detections):

        for class_id in detected:
            precision[class_id] = 0

        # get recall and confidence values
        for class_id in truth:
            precision[class_id] = 0
            
            if detected.get(class_id) is not None:
                score = detected[class_id].score

                truth_mask = truth[class_id].mask
                detected_mask = detected[class_id].mask

                if iou.get(class_id) is None:
                    iou[class_id] = [get_iou(truth_mask, detected_mask)]
                else:
                    iou[class_id].append(get_iou(truth_mask, detected_mask))


            else:
                score = 0

            if confidence.get(class_id) is None:
                confidence[class_id] = [score]
            else:
                confidence[class_id].append(score)
                
            if recall.get(class_id) is None and score != 0:
                recall[class_id] = 1
            
            elif recall.get(class_id) is not None and score !=0:
                recall[class_id] += 1
        
        # get precision values
        for class_id in detected:
            if truth.get(class_id) is None:
                precision[class_id] += 1


    return confidence, recall, precision, iou


def write_stats_to_json(confidence, recall, precision, iou, times, category_index, output_path):

    stats = defaultdict()
    stats['initialization_time'] = times[0]
    stats['inference_time_avg'] = round(sum(times[1:]) / len(times[1:]), 3)

    for class_id in category_index:

        if confidence.get(class_id) is not None:
            num_instances = len(confidence[class_id])
            avg_confidence = round(np.average(confidence[class_id]), 3)
            recall_stat = round(recall[class_id] / len(confidence[class_id]), 3)
            precision_stat = round(len(confidence[class_id]) / (len(confidence[class_id]) + precision[class_id]), 3)
            avg_iou = round(np.average(iou[class_id]), 3)

        else:
            num_instances = 0
            avg_confidence = 0.000
            recall_stat = 0.000
            precision_stat = 0.000
            avg_iou = 0.000
        
        class_stat = {"num_instances": num_instances,
                      "avg_confidence": avg_confidence,
                      "recall": recall_stat,
                      "precision": precision_stat,
                      "avg_iou": avg_iou}
        
        stats[class_id] = defaultdict()
        stats[class_id]['name'] = category_index[class_id]['name']
        stats[class_id]['class_stats'] = defaultdict()
        
        for key in class_stat:
            stats[class_id]['class_stats'][key] = class_stat[key]

    os.makedirs(Path(output_path), exist_ok=True)

    json_path = Path(output_path) / 'stats.json'
    with open(json_path, 'w') as fp:
        json.dump(stats, fp, indent=4)
