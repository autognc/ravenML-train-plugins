from matplotlib.colors import Normalize
import torch
import torchvision
from tfrecord.torch.dataset import MultiTFRecordDataset
import numpy as np
import subprocess
import os
import glob
import cv2
import torchvision.transforms as T

## TODO: ADD TRANSFORMS, load from directory
# Transforms: random flip, random crop, random contrast, color, jpeg, gaussian, albumentations augmentations
class HeatMapDataset(torch.utils.data.IterableDataset):

    def __init__(self, hp, data_dir, nb_keypoints, split_name):
        self.hp = hp
        self.data_dir = data_dir
        self.nb_keypoints = nb_keypoints
        split_prefix = f"{split_name}.record-"
        filenames = sorted(glob.glob(
            os.path.join(self.data_dir, split_prefix + "*" )
        ))
        record_pattern = os.path.join(self.data_dir, split_prefix + "{}" )
        index_pattern = os.path.join(self.data_dir, f"{split_name}.index-" + "{}" )
        splits = {}
        for filename in filenames:
            split_num = os.path.basename(filename)[len(split_prefix):]
            index_path = os.path.join(self.data_dir, f"{split_name}.index-{split_num}")
            if not os.path.exists(index_path):
                subprocess.call(["python", "-m", "tfrecord.tools.tfrecord2idx", filename, index_path])
            splits[split_num] = 1/len(filenames)
        features = {
            "image/height":"int",
            "image/width": "int",
            "image/object/keypoints": "float",
            "image/object/bbox/xmin":"float",
            "image/object/bbox/xmax": "float",
            "image/object/bbox/ymin": "float",
            "image/object/bbox/ymax": "float",
            "image/encoded": "byte",
            "image/object/pose": "float",
            "image/imageset": "byte",
        }
        self.dataset = iter(MultiTFRecordDataset(
            record_pattern, 
            index_pattern, 
            splits, 
            features,
            shuffle_queue_size=self.hp.get("shuffle_buffer_size", 1000),
            infinite=False
        ))
        self.heatmap_generator = [
            HeatmapGenerator(
                output_size, self.nb_keypoints, self.hp.dataset.sigma
            ) for output_size in self.hp.dataset.output_size
        ]
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Resize(self.hp.dataset.input_size)
        ])
    
    def __iter__(self):
        def iterator():
            for feat in self.dataset:
                keypoints = feat["image/object/keypoints"].reshape(-1, 2)[:self.nb_keypoints]
                image = cv2.cvtColor(cv2.imdecode(feat["image/encoded"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.uint8)
                heatmaps = [heatmap(keypoints)  for heatmap in self.heatmap_generator]
                image = self.transforms(image)
                yield image, heatmaps
        return iterator()
            





class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for idx, p in enumerate(joints):
            x, y = int(p[0]*self.output_res), int(p[1]*self.output_res)
            if x < 0 or y < 0 or \
                x >= self.output_res or y >= self.output_res:
                continue

            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

            cc, dd = max(0, ul[0]), min(br[0], self.output_res)
            aa, bb = max(0, ul[1]), min(br[1], self.output_res)
            hms[idx, aa:bb, cc:dd] = np.maximum(
                hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms
