import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from collections import defaultdict
from object_detection.metrics.coco_evaluation import CocoDetectionEvaluator
from object_detection.core.standard_fields import InputDataFields, DetectionResultFields
from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluator


class BoundingBoxEvaluator:

    def __init__(self, category_index, fov=None, distance_unit=None):
        """
        :param category_index: A category_index from a BoundingBoxModel. Can be retrieved with model.category_index.
        :param fov:  the FOV of the height or width axis of the camera used to take the images, in degrees. If this
            is provided, then `image_size` must also be provided to each call to `add_single_result` where `image_size`
            is the image resolution along the same dimension as `fov`. If these conditions are met, all distance
            statistics will also be computed in degrees. (optional)
        :param distance_unit: A string indicating what length unit `distance` will be in `add_single_result`
            (e.g. 'meters'). If this is provided, then `fov` must also be provided, and `distance`
            must be provided to every call to `add_single_result`. If these
            conditions are met, then all distance statistics will also be computed in these units.
        """
        if distance_unit and not fov:
            raise ValueError("distance_unit provided without fov.")
        self.category_index = category_index
        self.classes = [cls['name'] for cls in category_index.values()]
        self.count = 0
        self.times = []
        self.outputs = []
        self.bboxes = []
        self.centroids = []
        self.fov = fov
        self.distance_unit = distance_unit
        if fov:
            self.sizes = []
            if distance_unit:
                self.distances = []
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

    def add_single_result(self, output, true_shape, inference_time, bbox, centroid, image_size=None, distance=None):
        """
        Add single inference result to the evaluation.
        :param output: the parsed output from a BoundingBoxModel inference.
        :param inference_time: the inference time from a BoundingBoxModel inference.
        :param bbox: a dict {classname: bbox} where bbox has keys keys 'xmin', 'xmax', 'ymin',
            and 'ymax' in non-normalized (pixel) coordinates.
        :param centroid: a dict {classname: centroid} where classname is a tuple (y, x) in
            non-normalized (pixel) coordinates.
        :param image_size: the size of the image, in pixels, along the same dimension as the specified FOV in __init__.
        :param distance: a dict {classname: distance} where distance is the distance from the object
            to the camera in `distance_unit`.
        """
        output = self.parse_inference_output(output, true_shape)
        if image_size:
            if self.fov:
                self.sizes.append(image_size)
            else:
                raise ValueError("image_size provided without fov in __init__")
        elif self.fov:
            raise ValueError("image_size not provided when fov was provided in __init__")
        if distance:
            if self.distance_unit:
                self.distances.append(distance)
            else:
                raise ValueError("distance provided without distance_unit in __init__")
        elif self.distance_unit:
            raise ValueError("distance not provided when distance_unit was provided in __init__")

        print(f'Image {self.count}, time: {inference_time}')
        self.outputs.append(output)
        self.times.append(inference_time)
        self.bboxes.append(bbox)
        self.centroids.append(centroid)
        self.count += 1

    def parse_inference_output(self, output, image_size):
        """
        Parses the raw output of the object detection model into a more sensible format.
        :param category_index: a category index created with `get_category_index`
        :param output: a dict obtained by running the model and fetching, at the very least, all of the output tensors
        provided by `get_input_and_output_tensors`.
        :return: a dictionary of the form {classname: [(confidence, bbox)]} where bbox is a
        dict with keys xmin, xmax, ymin, ymax (non-normalized).
        """
        # unpack the outputs, which come with a batch dimension
        num_detections = int(output['num_detections'][0])
        detection_classes = output['detection_classes'][0].numpy() + 1
        detection_boxes = output['detection_boxes'][0].numpy()
        detection_scores = output['detection_scores'][0].numpy()
        image_size = image_size.numpy()

        detections = defaultdict(list)
        for i in range(num_detections):
            score = detection_scores[i]
            box = detection_boxes[i]
            class_id = int(detection_classes[i])
            class_name = self.category_index[class_id]['name']
            bbox = {
                'xmin': box[1] * image_size[0][1],
                'xmax': box[3] * image_size[0][1],
                'ymin': box[0] * image_size[0][0],
                'ymax': box[2] * image_size[0][0]
            }
            detections[class_name].append((score, bbox))

        return dict(detections)

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
        distances = {cls: {
            key: [] for key in self._gen_distance_keys(['error'])
        } for cls in self.classes}
        for i in range(len(self.bboxes)):
            for class_name in self.classes:
                bbox = self.bboxes[i].get(class_name)
                centroid = self.centroids[i].get(class_name)
                if not (bbox and centroid):
                    continue
                distances[class_name]['error_px'].append(self._get_distance(self._get_centroid(bbox), centroid))
                if self.fov:
                    distances[class_name]['error_deg'].append(
                        self._get_distance(self._get_centroid(bbox), centroid, self.fov / self.sizes[i])
                    )
                    if self.distance_unit:
                        distances[class_name][f'error_{self.distance_unit}'].append(
                            self._chord_length(distances[class_name]['error_deg'][-1], self.distances[i])
                        )
        for class_name, values in distances.items():
            for key, value in values.items():
                distances[class_name][key] = np.mean(np.array(value))

        if save:
            self.stats['avg_truth_bbox_to_truth_centroid_error'] = distances
        return distances

    def calculate_distance_statistics(self, confidence_threshold, save=True):
        stats = {cls: {
            key: [] for key in self._gen_distance_keys(['bbox_to_bbox', 'bbox_to_centroid'])
        } for cls in self.classes}
        for i in range(len(self.bboxes)):
            for class_name in self.classes:
                outputs = self.outputs[i][class_name]  # list of (score, bbox) for this class
                bbox = self.bboxes[i].get(class_name)
                centroid = self.centroids[i].get(class_name)
                # this will only consider true positives
                if not (bbox and centroid and outputs[0][0] >= confidence_threshold):
                    continue

                stats[class_name]['bbox_to_bbox_px'].append(
                    self._get_distance(self._get_centroid(outputs[0][1]), self._get_centroid(bbox))
                )
                stats[class_name]['bbox_to_centroid_px'].append(
                    self._get_distance(self._get_centroid(outputs[0][1]), centroid)
                )
                if self.fov:
                    deg_per_px = self.fov / self.sizes[i]
                    stats[class_name]['bbox_to_bbox_deg'].append(
                        self._get_distance(self._get_centroid(outputs[0][1]), self._get_centroid(bbox), deg_per_px)
                    )
                    stats[class_name]['bbox_to_centroid_deg'].append(
                        self._get_distance(self._get_centroid(outputs[0][1]), centroid, deg_per_px)
                    )
                    if self.distance_unit:
                        stats[class_name][f'bbox_to_bbox_{self.distance_unit}'].append(
                            self._chord_length(stats[class_name]['bbox_to_bbox_deg'][-1], self.distances[i])
                        )
                        stats[class_name][f'bbox_to_centroid_{self.distance_unit}'].append(
                            self._chord_length(stats[class_name]['bbox_to_centroid_deg'][-1], self.distances[i])
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

    def plot_dr_curve(self, save_dir='.', mode='px'):
        if mode not in ['px', 'deg', 'distance']:
            raise ValueError(f'mode {mode} not recognized')
        if mode == 'deg' and not self.fov:
            return
        if mode == 'distance' and not (self.fov and self.distance_unit):
            return
        unit = self.distance_unit if mode == 'distance' else mode
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
            distances = {k: [d[k] for d in distances] for k in distances[0].keys() if unit in k}
            for k, v in distances.items():
                ax.scatter(np.array(recalls), np.array(distances[k]), 10, label='_'.join(k.split('_')[:-1]))
            ax.set_title(f'Distance-Recall Curve ({unit}) for {class_name}')
            ax.set_xlabel('Recall')
            ax.set_ylabel(f'Distance Error ({unit})')
            plt.legend(title="Distance Type", loc='upper right')
            ax.set_xlim(0, 1)
            plt.savefig(os.path.join(save_dir, f'dr_{unit}_curve_{class_name}.png'))
            plt.clf()

    def plot_dt_curve(self, save_dir='.', mode='px'):
        if mode not in ['px', 'deg', 'distance']:
            raise ValueError(f'mode {mode} not recognized')
        if mode == 'deg' and not self.fov:
            return
        if mode == 'distance' and not (self.fov and self.distance_unit):
            return
        unit = self.distance_unit if mode == 'distance' else mode
        plt.clf()
        for class_name in self.classes:
            fig, ax = plt.subplots()
            distances = []
            scores = []
            for i in range(len(self.bboxes)):
                outputs = self.outputs[i][class_name]  # list of (score, bbox) for this class
                bbox = self.bboxes[i].get(class_name)
                centroid = self.centroids[i].get(class_name)
                if not (bbox and centroid):
                    distances.append(None)
                    scores.append(None)
                    continue
                scores.append(outputs[0][0])
                if mode == 'px':
                    distances.append(self._get_distance(self._get_centroid(outputs[0][1]), centroid))
                    continue
                angle = self._get_distance(self._get_centroid(outputs[0][1]), centroid, self.fov / self.sizes[i])
                if mode == 'deg':
                    distances.append(angle)
                elif mode == 'distance':
                    distances.append(self._chord_length(angle, self.distances[i]))
            ax.scatter(np.arange(len(distances)) + 1, distances, c=scores, cmap='viridis')
            ax.set_title(f'Distance Error (CoB-to-CoM, {unit}) vs Time for {class_name}')
            ax.set_xlabel('Image Number')
            if mode == 'distance':
                ax.set_ylim(0, 60)
            elif mode == 'deg':
                ax.set_ylim(0, 0.45)
            ax.set_ylabel(f'Distance Error ({unit})')
            ax.text(0.92, 0.9, 'Lighter = higher confidence', transform=ax.transAxes, horizontalalignment='right')
            plt.savefig(os.path.join(save_dir, f'dt_{unit}_curve_{class_name}.png'))
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

    def _gen_distance_keys(self, prefixes):
        keys = [prefix + '_px' for prefix in prefixes]
        if self.fov:
            keys += [prefix + '_deg' for prefix in prefixes]
        if self.distance_unit:
            keys += [prefix + '_' + self.distance_unit for prefix in prefixes]
        return keys

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
            rad_per_pixel = deg_per_pixel * np.pi / 180
            ap, al = ay * rad_per_pixel, ax * rad_per_pixel
            bp, bl = by * rad_per_pixel, bx * rad_per_pixel
            # Haversine formula for great circle distance
            return 2 * np.arcsin(np.sqrt(
                np.sin((ap - bp) / 2)**2 + np.cos(ap) * np.cos(bp) * np.sin((al - bl) / 2)**2)
            ) * 180 / np.pi
        return np.sqrt((ay - by) ** 2 + (ax - bx) ** 2)

    @classmethod
    def _chord_length(cls, angle, obj_distance):
        return 2 * obj_distance * np.sin(angle * np.pi / 180 / 2)

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
        self.plot_dr_curve(output_dir, mode='deg')
        self.plot_dr_curve(output_dir, mode='distance')
        self.plot_dt_curve(output_dir, mode='deg')
        self.plot_dt_curve(output_dir, mode='distance')
        self.plot_it_curve(output_dir)

        self.save_stats(os.path.join(output_dir, 'stats.json'))
