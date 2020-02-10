import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from object_detection.metrics.coco_evaluation import CocoDetectionEvaluator
from object_detection.core.standard_fields import InputDataFields, DetectionResultFields
from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluator


class BoundingBoxEvaluator:

    def __init__(self, category_index):
        """
        :param category_index: A category_index from a BoundingBoxModel. Can be retrieved with model.category_index.
        """
        self.category_index = category_index
        self.classes = [cls['name'] for cls in category_index.values()]
        self.count = 0
        self.times = []
        self.outputs = []
        self.bboxes = []
        self.centroids = []
        self.stats = {}

    @classmethod
    def load_from_dump(cls, dump_path):
        """
        Loads inference results from a previous dump, but not any statistics. Desired statistics must be
        recomputed from loaded results.
        """
        self = cls.__new__(cls)
        with open(dump_path, 'rb') as f:
            dump = pickle.load(f)
        for k, v in dump.items():
            vars(self)[k] = v
        self.stats = {}
        return self

    def add_single_result(self, output, inference_time, bbox, centroid):
        """
        Add single inference result to the evaluation.
        :param output: the parsed output from a BoundingBoxModel inference.
        :param inference_time: the inference time from a BoundingBoxModel inference.
        :param bbox: a dict with keys 'xmin', 'xmax', 'ymin', and 'ymax' in non-normalized (pixel) coordinates.
        :param centroid: (y, x) in non-normalized (pixel) coordinates.
        """
        print(f'Image {self.count}, time: {inference_time}')
        self.outputs.append(output)
        self.times.append(inference_time)
        self.bboxes.append(bbox)
        self.centroids.append(centroid)
        self.count += 1

    def calculate_coco_statistics(self, save=True):
        # create coco evaluator
        coco_evaluator = CocoDetectionEvaluator(list(self.category_index.values()))
        # OD evaluator (older than coco, I left it here in case we need it later)
        """
        od_evaluator = ObjectDetectionEvaluator(
            list(category_index.values()),
            matching_iou_threshold=iou_threshold,
            evaluate_corlocs=True,
            evaluate_precision_recall=True
        )
        """
        for i, (output_dict, bbox_dict) in enumerate(zip(self.outputs, self.bboxes)):
            for class_id, cls in self.category_index.items():
                class_name = cls['name']
                outputs = output_dict[class_name]  # list of (score, bbox) for this class
                bbox = bbox_dict.get(class_name)  # bbox for this class

                # add ground truth for this class and this image to the evaluator
                if bbox:
                    boxes = np.array([[bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax']]], dtype=np.float32)
                else:
                    boxes = np.empty([0, 4], dtype=np.float32)
                groundtruth_dict = {
                    InputDataFields.groundtruth_boxes: boxes,
                    InputDataFields.groundtruth_classes: np.full(1, class_id, dtype=np.float32)
                }
                coco_evaluator.add_single_ground_truth_image_info(i, groundtruth_dict)
                # od_evaluator.add_single_ground_truth_image_info(i, groundtruth_dict)

                # add detections for this class and this image to the evaluator
                scores, boxes = zip(*outputs)
                boxes = [[box['ymin'], box['xmin'], box['ymax'], box['xmax']] for box in boxes]
                detections_dict = {
                    DetectionResultFields.detection_boxes: np.array(boxes, dtype=np.float32),
                    DetectionResultFields.detection_scores: np.array(scores, dtype=np.float32),
                    DetectionResultFields.detection_classes: np.full(len(boxes), class_id, dtype=np.float32)
                }
                coco_evaluator.add_single_detected_image_info(i, detections_dict)
                # od_evaluator.add_single_detected_image_info(i, detections_dict)

        result = coco_evaluator.evaluate()
        if save:
            self.stats['coco_statistics'] = result
        return result

    def calculate_confusion_matrix(self, confidence_threshold, iou_threshold, save=True):
        # NOTE: normally, the way that a confusion matrix works for object detection is that the list of detections
        # for each image is filtered by the confidence threshold, and all detections that pass the threshold are
        # considered: i.e. it counts as a true positive if a lower-ranked detection matches the ground truth,
        # and any additional detections besides that first matching one count as false positives.
        # However, since we know that there's only one instance of each class, we're going to do things a bit
        # differently and only consider the top detection for each image, since this is more representative of
        # our actual application. Note that this means, in some cases, the recall will never reach 1.0 no matter
        # how low the confidence threshold is. The typical object detection scheme also makes it impossible
        # to distinguish between a false positive where the groundtruth box doesn't exist and a detection where
        # the groundtruth box exists but does not meet the IoU threshold. Since this distinction is very useful
        # in our single-instance detection scenario, we add a new entry into the matrix called 'misplaced_positive'
        # for situations where the groundtruth box exists and the top detection was over the confidence
        # threshold, but did not meet the IoU threshold.
        confusion_matrix = {cls: {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0,
            'misplaced_positive': 0
        } for cls in self.classes}
        for output_dict, bbox_dict in zip(self.outputs, self.bboxes):
            for class_name in self.classes:
                outputs = output_dict[class_name]  # list of (score, bbox) for this class
                bbox = bbox_dict.get(class_name)

                if outputs and outputs[0][0] >= confidence_threshold:
                    # the model made a detection, so check if it's correct
                    if bbox:
                        if self._get_iou(bbox, outputs[0][1]) >= iou_threshold:
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
        for class_name in self.classes:
            confusion_matrix[class_name]['precision'] = self._get_precision(confusion_matrix[class_name])
            confusion_matrix[class_name]['recall'] = self._get_recall(confusion_matrix[class_name])
        if save:
            self.stats[f'confusion_matrix@{confidence_threshold}c,{iou_threshold}iou'] = confusion_matrix
        return confusion_matrix

    def calculate_truth_bbox_to_truth_centroid_error(self, save=True):
        distances = {cls: [] for cls in self.classes}
        for bbox_dict, centroid_dict in zip(self.bboxes, self.centroids):
            for class_name in self.classes:
                bbox = bbox_dict.get(class_name)
                centroid = centroid_dict.get(class_name)
                if not (bbox and centroid):
                    continue
                distances[class_name].append(self._get_distance(self._get_centroid(bbox), centroid))
        for k, v in distances.items():
            distances[k] = np.mean(np.array(distances[k]))

        if save:
            self.stats['avg_truth_bbox_to_truth_centroid_error'] = distances
        return distances

    def calculate_distance_statistics(self, confidence_threshold, save=True):
        stats = {cls: {
            'bbox_to_bbox': [],
            'bbox_to_centroid': []
        } for cls in self.classes}
        for output_dict, bbox_dict, centroid_dict in zip(self.outputs, self.bboxes, self.centroids):
            for class_name in self.classes:
                outputs = output_dict[class_name]  # list of (score, bbox) for this class
                bbox = bbox_dict.get(class_name)
                centroid = centroid_dict.get(class_name)
                # this will only consider true positives
                if not (bbox and centroid and outputs[0][0] >= confidence_threshold):
                    continue

                stats[class_name]['bbox_to_bbox'].append(
                    self._get_distance(self._get_centroid(outputs[0][1]), self._get_centroid(bbox))
                )
                stats[class_name]['bbox_to_centroid'].append(
                    self._get_distance(self._get_centroid(outputs[0][1]), centroid)
                )

        # average everything
        for class_name, values in stats.items():
            for key, value in values.items():
                stats[class_name][key] = np.mean(np.array(value))

        if save:
            self.stats[f'avg_distances@{confidence_threshold}c'] = stats
        return stats

    def plot_pr_curve(self, save_dir='.', iou_thresholds=(0.1, 0.25, 0.5, 0.75, 0.9)):
        plt.clf()
        for class_name in self.classes:
            fig, ax = plt.subplots()
            for iou_threshold in iou_thresholds:
                precisions = []
                recalls = []
                for score in np.arange(0, 1.025, 0.025)[::-1]:
                    cf = self.calculate_confusion_matrix(score, iou_threshold, save=False)[class_name]
                    recalls.append(cf['recall'])
                    precisions.append(cf['precision'])
                ax.scatter(np.array(recalls), np.array(precisions), 10, label=str(iou_threshold))
            ax.set_title(f'PR Curve for {class_name}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            plt.legend(title="IOU Threshold", loc='upper left')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            plt.savefig(os.path.join(save_dir, f'pr_curve_{class_name}.png'))
            plt.clf()

    def plot_dr_curve(self, save_dir='.'):
        plt.clf()
        for class_name in self.classes:
            fig, ax = plt.subplots()
            recalls = []
            distances = []
            for score in np.arange(0, 1.025, 0.025)[::-1]:
                # get recall ignoring IOU (meaning all detections when a truth bbox exists somewhere in the image
                # are counted as true positive)
                cf = self.calculate_confusion_matrix(score, 0.0, save=False)[class_name]
                recalls.append(self._get_recall(cf))
                # get average distances
                distances.append(
                    self.calculate_distance_statistics(score, save=False)[class_name]
                )
            distances = {k: [d[k] for d in distances] for k in distances[0].keys()}
            for k, v in distances.items():
                ax.scatter(np.array(recalls), np.array(distances[k]), 10, label=k)
            ax.set_title(f'DR Curve for {class_name}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Distance Error')
            plt.legend(title="Distance Type", loc='upper right')
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            plt.savefig(os.path.join(save_dir, f'dr_curve_{class_name}.png'))
            plt.clf()

    def plot_dt_curve(self, save_dir='.'):
        plt.clf()
        for class_name in self.classes:
            fig, ax = plt.subplots()
            distances = []
            scores = []
            for output_dict, bbox_dict, centroid_dict in zip(self.outputs, self.bboxes, self.centroids):
                outputs = output_dict[class_name]  # list of (score, bbox) for this class
                bbox = bbox_dict.get(class_name)
                centroid = centroid_dict.get(class_name)
                if not (bbox and centroid):
                    distances.append(None)
                    scores.append(None)
                    continue
                distances.append(self._get_distance(self._get_centroid(outputs[0][1]), centroid))
                scores.append(outputs[0][0])
            ax.scatter(np.arange(len(distances)) + 1, distances, c=scores, cmap='viridis')
            ax.set_title(f'Distance Error vs Time for {class_name}')
            ax.set_xlabel('Image Number')
            ax.set_ylabel('Distance Error (deg)')
            ax.text(0.92, 0.9, 'Lighter = higher confidence', transform=ax.transAxes, horizontalalignment='right')
            plt.savefig(os.path.join(save_dir, f'dt_curve_{class_name}.png'))
            plt.clf()

    def plot_it_curve(self, save_dir='.'):
        plt.clf()
        for class_name in self.classes:
            fig, ax = plt.subplots()
            distances = []
            scores = []
            for output_dict, bbox_dict, centroid_dict in zip(self.outputs, self.bboxes, self.centroids):
                outputs = output_dict[class_name]  # list of (score, bbox) for this class
                bbox = bbox_dict.get(class_name)
                centroid = centroid_dict.get(class_name)
                if not (bbox and centroid):
                    distances.append(None)
                    scores.append(None)
                    continue
                distances.append(self._get_iou(outputs[0][1], bbox))
                scores.append(outputs[0][0])
            ax.scatter(np.arange(len(distances)) + 1, distances, c=scores, cmap='viridis')
            ax.set_title(f'IoU vs Time for {class_name}')
            ax.set_xlabel('Image Number')
            ax.set_ylabel('IoU')
            ax.set_ylim(0, 1)
            ax.text(0.92, 0.9, 'Lighter = higher confidence', transform=ax.transAxes, horizontalalignment='right')
            plt.savefig(os.path.join(save_dir, f'it_curve_{class_name}.png'))
            plt.clf()

    def save_stats(self, path):
        self.stats['average_inference_time'] = np.mean(np.array(self.times))
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)

    def dump(self, path):
        """Dumps accumulated inference results to a pickle file, but not any statistics computed from the results"""
        results = {k: v for k, v in vars(self).items() if k != 'stats'}
        with open(path, 'wb') as f:
            pickle.dump(results, f)

    @classmethod
    def _get_area(cls, a):
        return (a['xmax'] - a['xmin']) * (a['ymax'] - a['ymin'])

    @classmethod
    def _get_iou(cls, a, b):
        intersection = {
            'xmin': max(a['xmin'], b['xmin']),
            'ymin': max(a['ymin'], b['ymin']),
            'xmax': min(a['xmax'], b['xmax']),
            'ymax': min(a['ymax'], b['ymax'])
        }
        intersection_area = cls._get_area(intersection)
        union_area = cls._get_area(a) + cls._get_area(b) - intersection_area
        return intersection_area / union_area

    @classmethod
    def _get_precision(cls, cf):
        if cf['true_positive'] + cf['false_positive'] + cf['misplaced_positive'] > 0:
            return cf['true_positive'] / (cf['true_positive'] + cf['false_positive'] + cf['misplaced_positive'])
        return 0

    @classmethod
    def _get_recall(cls, cf):
        if cf['true_positive'] + cf['false_negative'] + cf['misplaced_positive'] > 0:
            return cf['true_positive'] / (cf['true_positive'] + cf['false_negative'] + cf['misplaced_positive'])
        return 0

    @classmethod
    def _get_centroid(cls, bbox):
        return (bbox['ymin'] + bbox['ymax']) / 2, (bbox['xmin'] + bbox['xmax']) / 2

    @classmethod
    def _get_distance(cls, centroid_a, centroid_b, deg_per_pixel=None):
        """
        If deg_per_pixel is not provided or None, returns the Euclidean pixel distance.
        If deg_per_pixel is provided, it is used to convert the pixel coordinates to
        azimuth and elevation, and then returns the great circle distance.
        """
        ay, ax = centroid_a
        by, bx = centroid_b

        if deg_per_pixel:
            ap, al = ay * deg_per_pixel, ax * deg_per_pixel
            bp, bl = by * deg_per_pixel, bx * deg_per_pixel
            # Haversine formula for great circle distance
            return 2 * np.arcsin(np.sqrt(
                np.sin((ap - bp) / 2)**2 + np.cos(ap) * np.cos(bp) * np.sin((al - bl) / 2)**2)
            )
        return np.sqrt((ay - by) ** 2 + (ax - bx) ** 2)

    def calculate_default_and_save(self, output_dir):
        """Convenient method to calculate a bunch of default statistics and save them"""
        self.calculate_coco_statistics()
        self.calculate_truth_bbox_to_truth_centroid_error()

        self.calculate_confusion_matrix(0, 0.1)
        self.calculate_confusion_matrix(0, 0.5)
        self.calculate_confusion_matrix(0.3, 0.1)
        self.calculate_confusion_matrix(0.3, 0.5)
        self.calculate_confusion_matrix(0.7, 0.1)
        self.calculate_confusion_matrix(0.7, 0.5)

        self.calculate_distance_statistics(0.3)
        self.calculate_distance_statistics(0.5)
        self.calculate_distance_statistics(0.75)

        self.plot_pr_curve(output_dir)
        self.plot_dr_curve(output_dir)
        self.plot_dt_curve(output_dir)
        self.plot_it_curve(output_dir)

        self.save_stats(os.path.join(output_dir, 'stats.json'))
