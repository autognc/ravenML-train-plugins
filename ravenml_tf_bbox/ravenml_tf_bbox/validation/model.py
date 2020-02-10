import tensorflow as tf
import numpy as np
import time
from contextlib import contextmanager
from collections import defaultdict
from object_detection.utils import label_map_util
import object_detection.utils.visualization_utils as visualization


class BoundingBoxModel:

    def __init__(self, model_path, label_map_path):
        self.graph = self.get_model_graph(model_path)
        self.category_index = self.get_category_index(label_map_path)
        self.input_tensor, self.output_tensors = self.get_input_and_output_tensors(self.graph)
        self.sess = None

    @contextmanager
    def start_session(self, **kwargs):
        self.sess = tf.Session(graph=self.graph, **kwargs)
        try:
            yield None
        finally:
            self.sess.close()
            self.sess = None

    def run_inference_on_single_image(self, image, vis=False, vis_threshold=0.0):
        """
        Must be called inside a "with start_session():" block.

        :param image: NumPy array, shape (h, w, 3)
        :param vis: Whether or not to draw the top detection for each class on top of the input image and return it.
        :param vis_threshold: Only has an effect if vis is true. If the top detection's confidence score is
            below this threshold for a class, no bounding box will be drawn.
        :return: A tuple (output, inference_time). If vis is true, a tuple (output, inference_time, image).
        """
        if not self.sess:
            raise ValueError('Please call this method inside of a "with start_session():" block')
        start_time = time.time()
        raw_output = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: image[None, ...]})
        inference_time = time.time() - start_time
        parsed = self.parse_inference_output(self.category_index, raw_output, image.shape[0], image.shape[1])
        if vis:
            for class_name, detections in parsed.items():
                score, bbox = detections[0]  # only take top detection
                if score >= vis_threshold:
                    visualization.draw_bounding_box_on_image_array(
                        image, bbox['ymin'], bbox['xmin'], bbox['ymax'], bbox['xmax'],
                        color='green', thickness=1, display_str_list=[f'{class_name}: {int(score * 100)}%'],
                        use_normalized_coordinates=False
                    )
            return parsed, inference_time, image
        return parsed, inference_time

    @classmethod
    def get_num_classes(cls, label_path):
        with open(label_path, "r") as f:
            ids = [line for line in f if "id:" in line]
            num_classes = len(ids)
        return num_classes

    @classmethod
    def get_category_index(cls, label_path: str):
        label_map = label_map_util.load_labelmap(label_path)
        num_classes = cls.get_num_classes(label_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    @classmethod
    def get_model_graph(cls, model_path):
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph

    @classmethod
    def get_input_and_output_tensors(cls, graph):
        """
        Returns (image_tensor, output_tensors) where image_tensor is a placeholder to be used in feed_dict and
        output_tensors is a dictionary of fetchable output tensors including num_detections, detection_boxes,
        detection_scores, and detection_classes.
        """
        output_tensors = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes']:
            tensor_name = key + ':0'
            output_tensors[key] = graph.get_tensor_by_name(tensor_name)
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        return image_tensor, output_tensors

    @classmethod
    def parse_inference_output(cls, category_index, output, image_height, image_width):
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
        detection_classes = output['detection_classes'][0].astype(np.int)
        detection_boxes = output['detection_boxes'][0]
        detection_scores = output['detection_scores'][0]

        detections = defaultdict(list)
        for i in range(num_detections):
            score = detection_scores[i]
            box = detection_boxes[i]
            class_id = detection_classes[i]
            class_name = category_index[class_id]['name']
            bbox = {
                'xmin': box[1] * image_width,
                'xmax': box[3] * image_width,
                'ymin': box[0] * image_height,
                'ymax': box[2] * image_height
            }
            detections[class_name].append((score, bbox))

        return dict(detections)
