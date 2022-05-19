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
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import json
import pickle

## TODO: ADD TRANSFORMS, load from directory
# Transforms: random flip, random crop, random contrast, color, jpeg, gaussian, albumentations augmentations

# class TFDataset(Dataset):



class HeatMapDataset(Dataset):

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        meta_filename = self.meta_filenames[idx]
        with open(meta_filename, 'r') as f:
            metadata = json.load(f)
        # image, heatmaps, imdims, pose, translation 
        return self.get_transforms(metadata, image_filename)
        

    def get_transforms(self, feat, image_filename):
        # denormalize keypoint values
        imdims = np.array([feat["image/height"], feat["image/width"]])
        keypoints = imdims.T * np.array(feat["image/object/keypoints"]).reshape(-1, 2)[:self.nb_keypoints] ## convert keypoints to pixel space
        keypoints[:,0] = np.clip(keypoints[:,0], 0, imdims[0]-0.01)
        keypoints[:,1] = np.clip(keypoints[:,1], 0, imdims[1]-0.01)
        image = cv2.cvtColor(cv2.imread(image_filename), cv2.COLOR_BGR2RGB).astype(np.uint8)
        transformed = self.transforms(image=image, keypoints=keypoints) # apply transformations
        keypoints = np.array(transformed["keypoints"]) / self.hp.dataset.input_size # renormalize keypoints
        heatmaps = [heatmap(keypoints) for heatmap in self.heatmap_generator] # generate groundtruth heatmaps 
        return transformed["image"], heatmaps, imdims, np.array(feat["image/object/pose"]), np.array(feat["image/object/translation"])

    
    def _get_train_transforms(self):

        train_transforms = []
        train_transforms.append(
            A.RandomRotate90(p=1.0)
        )
        train_transforms.append(
            A.ColorJitter(
                brightness=self.hp.get("random_brightness", 0),
                contrast=self.hp.get("random_contrast", 0),
                saturation=self.hp.get("random_saturation", 0),
                hue=self.hp.get("random_hue", 0),
            )
        )

        if "random_gaussian" in self.hp:
            train_transforms.append(A.GaussNoise(self.hp["random_gaussian"]))

        if "random_jpeg" in self.hp:
            train_transforms.append(A.ImageCompression(*self.hp["random_jpeg"], always_apply=True))
        if "random_fog" in self.hp:
            train_transforms.append(A.RandomFog(p=self.hp["random_fog"]))
        if "random_shadow" in self.hp:
            train_transforms.append(A.RandomShadow(p=self.hp["random_shadow"]))
        if "random_rain" in self.hp:
            train_transforms.append(A.RandomRain(p=self.hp["random_rain"]))
        if "random_sun_flare" in self.hp:
            train_transforms.append(A.RandomSunFlare(src_radius=200,p=self.hp["random_sun_flare"]))
        if "random_snow" in self.hp:
            train_transforms.append(A.RandomSnow(p=self.hp["random_snow"]))
    
        return train_transforms

    def __init__(self, hp, data_dir, nb_keypoints, split_name):
        # with open('hp', 'wb') as f:
        #     pickle.dump(hp, f)
        # input()
        self.hp = hp
        self.data_dir = data_dir
        self.nb_keypoints = nb_keypoints
        self.split_name = split_name
        split_prefix = f"{split_name}/"
        self.image_filenames = sorted(glob.glob(
            os.path.join(self.data_dir, split_prefix, "img*jpg")
        ))[:1000]
        self.meta_filenames = sorted(glob.glob(
            os.path.join(self.data_dir, split_prefix, "img*.json")
        ))[:1000]
        self.heatmap_generator = [
            HeatmapGenerator(
                output_size, self.nb_keypoints, self.hp.dataset.sigma
            ) for output_size in self.hp.dataset.output_size
        ]
        self.transforms =[     
            # these values come from the imagenet dataset
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            A.Resize(self.hp.dataset.input_size, self.hp.dataset.input_size),
        ]
        if self.split_name == "train":
            self.transforms.extend(
                self._get_train_transforms()
            )
        self.transforms = A.Compose(
            [*self.transforms, ToTensorV2()],
            keypoint_params=A.KeypointParams(format='yx')
        )


class HeatMapDataset2():

    def __init__(self, hp, data_dir, nb_keypoints, split_name):
        self.hp = hp
        self.data_dir = data_dir
        self.nb_keypoints = nb_keypoints
        self.split_name = split_name
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
            "image/object/translation": "float",
            "image/filename": "byte"
        }
        self.dataset = MultiTFRecordDataset(
            record_pattern, 
            index_pattern, 
            splits, 
            features,
            shuffle_queue_size=self.hp.get("shuffle_buffer_size", 1000) if self.split_name == "train" else 0,
            infinite=False,
            transform=self._tfrecord_transforms
        )
        self.heatmap_generator = [
            HeatmapGenerator(
                output_size, self.nb_keypoints, self.hp.dataset.sigma
            ) for output_size in self.hp.dataset.output_size
        ]
        self.transforms =[     
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Resize(self.hp.dataset.input_size, self.hp.dataset.input_size),
        ]
        if self.split_name == "train":
            self.transforms.extend(
                self._get_train_transforms()
            )
        self.transforms = A.Compose(
            [*self.transforms, ToTensorV2()],
            keypoint_params=A.KeypointParams(format='yx')
        )
        
    def _tfrecord_transforms(self, feat):
        imdims = np.array([feat["image/height"], feat["image/width"]])
        keypoints = imdims.T * feat["image/object/keypoints"].reshape(-1, 2)[:self.nb_keypoints] ## convert keypoints to pixel space
        image = cv2.cvtColor(cv2.imdecode(feat["image/encoded"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.uint8)
        transformed = self.transforms(image=image, keypoints=keypoints) # apply transformations
        keypoints = np.array(transformed["keypoints"]) / self.hp.dataset.input_size # renormalize keypoints
        heatmaps = [heatmap(keypoints) for heatmap in self.heatmap_generator] # generate groundtruth heatmaps 
        return transformed["image"], heatmaps, imdims, feat["image/object/pose"], feat["image/object/translation"]

    
    def _get_train_transforms(self):

        train_transforms = []
        train_transforms.append(
            A.RandomRotate90(p=1.0)
        )
        train_transforms.append(
            A.ColorJitter(
                brightness=self.hp.get("random_brightness", 0),
                contrast=self.hp.get("random_contrast", 0),
                saturation=self.hp.get("random_saturation", 0),
                hue=self.hp.get("random_hue", 0),
            )
        )

        if "random_gaussian" in self.hp:
            train_transforms.append(A.GaussNoise(self.hp["random_gaussian"]))

        if "random_jpeg" in self.hp:
            train_transforms.append(A.ImageCompression(*self.hp["random_jpeg"], always_apply=True))
        if "random_fog" in self.hp:
            train_transforms.append(A.RandomFog(p=self.hp["random_fog"]))
        if "random_shadow" in self.hp:
            train_transforms.append(A.RandomShadow(p=self.hp["random_shadow"]))
        if "random_rain" in self.hp:
            train_transforms.append(A.RandomRain(p=self.hp["random_rain"]))
        if "random_sun_flare" in self.hp:
            train_transforms.append(A.RandomSunFlare(src_radius=200,p=self.hp["random_sun_flare"]))
        if "random_snow" in self.hp:
            train_transforms.append(A.RandomSnow(p=self.hp["random_snow"]))
    
        return train_transforms

        

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
            x, y = int(p[1]*self.output_res), int(p[0]*self.output_res)
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
        return hms + np.random.normal(loc=0, scale=0.01, size=hms.shape)
