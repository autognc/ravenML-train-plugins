import numpy as np
from collections import defaultdict
import json
import os
from pathlib import Path
import math

def get_iou(truth_mask, detected_mask):
    truth_mask = np.array(truth_mask, dtype=bool)
    detected_mask = np.array(detected_mask, dtype=bool)

    intersection = truth_mask * detected_mask
    union = truth_mask + detected_mask

    iou = round(intersection.sum()/float(union.sum()), 3)

    return iou

def get_solar_iou(solar_truth, solar_detected):

    t_h, t_w = solar_truth[0].mask.shape[:2]
    d_h, d_w = solar_detected[0].mask.shape[:2]
    
    if len(solar_detected) == 2 and len(solar_truth) == 2:
        ious = []
        for d in solar_detected:
            temp = []
            for t in solar_truth:
                iou = get_iou(t.mask, d.mask)
                temp.append(iou)

            ious.append(max(temp))

        return round(np.average(ious), 3)

    elif len(solar_detected) == 2 and len(solar_truth) == 1:
        ious = []
        t = solar_truth[0]
        for d in solar_detected:
            iou = get_iou(t.mask, d.mask)
            ious.append(iou)

        return max(ious)

    elif len(solar_detected) == 1:
        ious = []
        d = solar_detected[0]
        for t in solar_truth:
            iou = get_iou(t.mask, d.mask)
            ious.append(iou)

        return max(ious)

def get_scaled_dist(truth_cent, detected_cent, detected_size):
    d_h, d_w = detected_size

    delta_y = (truth_cent[0] - detected_cent[0]) / d_h
    delta_x = (truth_cent[1] - detected_cent[1]) / d_w
    
    return round(math.sqrt( (delta_y**2)+(delta_x**2) ), 3)

def get_dist(truth_cent, detected_cent):
    delta_y = (truth_cent[0] - detected_cent[0])
    delta_x = (truth_cent[1] - detected_cent[1])
    
    return round(math.sqrt( (delta_y**2)+(delta_x**2) ), 3)

def get_solar_centroid_distance(solar_truth, solar_detected):
    if len(solar_detected) == 2 and len(solar_truth) == 2:
        dists = []
        for d in solar_detected:
            temp = []
            for t in solar_truth:
                dist = get_dist(t.centroid, d.centroid)
                temp.append(dist)

            dists.append(max(temp))

        return round(np.average(dists), 3)
        
    elif len(solar_detected) == 2 and len(solar_truth) == 1:
        dists = []
        t = solar_truth[0]
        for d in solar_detected:
            dist = get_dist(t.centroid, d.centroid)
            dists.append(dist)

        return max(dists)
    
    elif len(solar_detected) == 1:
        dists = []
        d = solar_detected[0]
        for t in solar_truth:
            dist = get_dist(t.centroid, d.centroid)
            dists.append(dist)

        return max(dists)

def get_scaled_solar_centroid_distance(solar_truth, solar_detected):
    if len(solar_detected) == 2 and len(solar_truth) == 2:
        dists = []
        for d in solar_detected:
            temp = []
            for t in solar_truth:
                dist = get_scaled_dist(t.centroid, d.centroid, d.mask.shape[:2])
                temp.append(dist)

            dists.append(max(temp))

        return round(np.average(dists), 3)
        
    elif len(solar_detected) == 2 and len(solar_truth) == 1:
        dists = []
        t = solar_truth[0]
        for d in solar_detected:
            dist = get_scaled_dist(t.centroid, d.centroid, d.mask.shape[:2])
            dists.append(dist)

        return max(dists)
    
    elif len(solar_detected) == 1:
        dists = []
        d = solar_detected[0]
        for t in solar_truth:
            dist = get_scaled_dist(t.centroid, d.centroid, d.mask.shape[:2])
            dists.append(dist)

        return max(dists)


def calculate_statistics(all_truths, all_detections, category_index):
    confidence = defaultdict()
    accuracy = defaultdict()
    recall = defaultdict()
    precision = defaultdict()
    iou = defaultdict()
    parameters = defaultdict()
    centroid_dists = defaultdict()
    scaled_centroid_dists = defaultdict()

    for class_id in category_index:
        iou[class_id] = []
        confidence[class_id] = []
        centroid_dists[class_id] = []
        scaled_centroid_dists[class_id] = []
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        if category_index[class_id]['name'] == 'solar_panel':
            detected_solar_panels = 0
            truth_solar_panels = 0

            for truth, detected in zip(all_truths, all_detections):
                # true positive, we want high scores, to compute iou, and centroid dist
                if class_id in detected and class_id in truth:
                    TP += 1

                    iou[class_id].append(get_solar_iou(truth[class_id], detected[class_id]))
                    centroid_dists[class_id].append(get_solar_centroid_distance(truth[class_id], detected[class_id]))
                    scaled_centroid_dists[class_id].append(get_scaled_solar_centroid_distance(truth[class_id], detected[class_id]))

                    for d in detected[class_id]:
                        confidence[class_id].append(d.score)

                    detected_solar_panels += len(detected[class_id])
                    truth_solar_panels += len(truth[class_id])

                # false positive, we want low scores
                elif class_id in detected and class_id not in truth:
                    FP += 1
                    confidence[class_id].append(100.000 - detected[class_id].score)
                
                # false negative, do we want score = 0?? idk
                elif class_id not in detected and class_id in truth:
                    FN += 1

                    truth_solar_panels += len(truth[class_id])

                # true negative, do we want score = 100?? idk
                else:
                    TN += 1

        else:

            for truth, detected in zip(all_truths, all_detections):
                # true positive, we want high scores, to compute iou, and centroid dist
                if class_id in detected and class_id in truth:
                    TP += 1

                    truth_mask = truth[class_id].mask
                    detected_mask = detected[class_id].mask
                    iou[class_id].append(get_iou(truth_mask, detected_mask))

                    truth_centroid = truth[class_id].centroid
                    if truth_centroid is not None:
                        detected_centroid = detected[class_id].centroid
                        scaled_centroid_dists[class_id].append(get_scaled_dist(truth_centroid, detected_centroid, detected[class_id].mask.shape[:2]))
                        centroid_dists[class_id].append(get_dist(truth_centroid, detected_centroid))

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


    return confidence, accuracy, recall, precision, iou, parameters, centroid_dists, scaled_centroid_dists, detected_solar_panels, truth_solar_panels


def write_stats_to_json(confidence, accuracy, recall, precision, iou, parameters, centroid_dists, scaled_centroid_dists, \
                        detected_solar_panels, truth_solar_panels, \
                        times, category_index, output_path):

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

        if centroid_dists.get(class_id) is not None:
            avg_dist = round(np.average(centroid_dists[class_id]), 3)
            avg_scaled_dist = round(np.average(scaled_centroid_dists[class_id]), 3)
        else:
            avg_dist = 0.000
            avg_scaled_dist = 0.000

        class_stat = {"num_instances": num_instances,
                      "avg_confidence": avg_confidence,
                      "accuracy": acc_stat,
                      "recall": recall_stat,
                      "precision": precision_stat,
                      "avg_iou": avg_iou,
                      "avg_centroid_dist": avg_dist,
                      "avg_scaled_centroid_dist": avg_scaled_dist}

        if category_index[class_id]['name'] == 'solar_panel':
            class_stat['num_detected_solar_panels'] = detected_solar_panels
            class_stat['num_truth_solar_panels'] = truth_solar_panels

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
