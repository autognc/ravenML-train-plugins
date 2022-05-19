import numpy as np
import traceback

import os
from . import utils
import cv2
import torch
import torchvision
import pytorch_lightning as pl
from rmltrainpthrnet.model import get_pose_net
from rmltrainpthrnet.utils.dataset import HeatMapDataset
from rmltrainpthrnet.utils.heatmaps import aggregate_heatmaps, keypoints_from_output
import rmltrainpthrnet.utils.pose as pose_utils
import torch.nn as nn
import torch.optim as optim
import math
import random
import matplotlib.pyplot as plt
## define trainer here

## TODO: optimizer options, model options, validation from directory, val step pose error, comet logging
class HRNET(pl.LightningModule):

    def __init__(self, hp, data_dir, artifact_dir, keypoints_3d, experiment=None):
        super().__init__()
        self.hp = hp
        self.nb_keypoints = hp["keypoints"]
        self.data_dir = data_dir
        self.artifact_dir = artifact_dir
        self.keypoints_3d = keypoints_3d[: self.nb_keypoints]
        self.focal_length = self.hp["pnp_focal_length"]
        self.model = get_pose_net(self.hp, True)
        self.loss = MultiHeatMapLoss(self.nb_keypoints)
        self.experiment = experiment
    
    def forward(self, images):  
        return self.model(images)
    
    def train_dataloader(self):
        ds = HeatMapDataset(self.hp, self.data_dir, self.nb_keypoints, "train")
        loader = torch.utils.data.DataLoader(ds, num_workers=self.hp.dataset.num_workers, batch_size=self.hp["batch_size"])
        #loader = torch.utils.data.DataLoader(ds, batch_size=self.hp["batch_size"])
        return loader

    def val_dataloader(self):
        ds = HeatMapDataset(self.hp, self.data_dir, self.nb_keypoints, "test")
        loader = torch.utils.data.DataLoader(ds, num_workers=self.hp.dataset.num_workers, batch_size=1)
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp["learning_rate"]) ##TODO: more optimizer options
        """lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, 
                    block["exp"], 
                    last_epoch= -1 if i == 0 else self.hp["lr_schedule"][i-1]["epoch"]
                ) 
                for i, block in enumerate(self.hp["lr_schedule"])
            ]
        )"""
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100) ##TODO: better LR schedule
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        images, heatmaps, _,_,_ = batch
        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        y_hat = self.model(images)
        losses = self.loss(y_hat, heatmaps)
        total_loss = 0
        for loss in losses:
            total_loss += loss.mean(dim=0)
        self.log("train_loss", total_loss)
        #print(self.trainer.lr_schedulers[0]["scheduler"].get_last_lr()[0])
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        images, heatmaps, imdims, pose, translation = batch
        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        heatmap_pred = self.model(images)
        losses = self.loss(heatmap_pred, heatmaps)
        total_loss = 0
        for loss in losses:
            total_loss += loss.mean(dim=0)
        self.log("validation_Loss", total_loss)
        
        keypoints_batch = keypoints_from_output(heatmap_pred, self.hp.dataset.input_size, 20).cpu().numpy()
        keypoints_batch = keypoints_batch.transpose(0,2,1,3)
        pose_errs = []
        trans_errs = []
        for i, kps in enumerate(keypoints_batch):
            try:
                imdims_single = imdims[i].cpu().numpy()
                kps_norm = kps
                kps = (kps*imdims_single.T)/self.hp.dataset.input_size
                r_vec, t_vec = pose_utils.solve_pose(
                        self.keypoints_3d,
                        kps,
                        [self.focal_length, self.focal_length],
                        imdims_single,
                        ransac=True,
                        reduce_mean=False,
                )
                pose_errs.append(pose_utils.geodesic_error(r_vec, pose[i].cpu().numpy()))
                trans_errs.append(pose_utils.position_error(t_vec, translation[i].cpu().numpy())[1])
                if random.random() < 0.01:
                    image = images[i]
                    for t, m, s in zip(image, [0.485, 0.456, 0.406], (0.229, 0.224, 0.225)):
                        t.mul_(s).add_(m)
                    image = image*255
                    num = random.randint(0,100)
                    cv2.imwrite(f"{self.current_epoch}_{num:03d}_eval.png", vis_keypoints(cv2.cvtColor(image.cpu().numpy().transpose(1,2,0), cv2.COLOR_RGB2BGR), kps_norm, pose_errs[-1], trans_errs[-1]))
                    
                    hm = aggregate_heatmaps(heatmap_pred, self.hp.dataset.input_size).cpu().numpy()[0].sum(axis=0)
                    plt.imsave(f"{self.current_epoch}_{num:03d}_eval_hm.png", hm)
                    hm = aggregate_heatmaps(heatmaps, self.hp.dataset.input_size).cpu().numpy()[0].sum(axis=0)
                    plt.imsave(f"{self.current_epoch}_{num:03d}_eval_gt.png", hm)
            except:
                pass
            
            

        self.log("validation_pose_err", np.degrees(np.mean(pose_errs)), prog_bar=True)
        self.log("validation_translation_err", np.mean(trans_errs), prog_bar=True)    
        return total_loss

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2)
        loss = loss.mean()#loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


class MultiHeatMapLoss(nn.Module):

    def __init__(self, nb_keypoints, loss_factor=1.0):
        super().__init__()
        self.nb_keypoints = nb_keypoints
        self.loss_factor = loss_factor
        self.heatmap_losses =\
            nn.ModuleList(
                [
                    HeatmapLoss()
                    for _ in range(self.nb_keypoints)
                ]
            )
        
    
    def forward(self, outputs, heatmaps):
        heatmaps_losses = []
        for i in range(len(outputs)):
            heatmaps_pred = outputs[i][:, :self.nb_keypoints]
            heatmaps_loss = self.heatmap_losses[i](
                    heatmaps_pred, heatmaps[i]
            )
            heatmaps_loss = heatmaps_loss * self.loss_factor
            heatmaps_losses.append(heatmaps_loss)

        return heatmaps_losses



def vis_keypoints(image, kps, err_rot=None, err_pos=None):
    """
    :param kps: shape (nb_keypoints, guesses_per_kp, 2)
    """
    nb_keypoints = kps.shape[1]
    kps_by_kp = kps.transpose([1, 0, 2])
    hues = np.linspace(0, 360, num=nb_keypoints, endpoint=False, dtype=np.float32)
    colors = np.stack(
        [
            hues,
            np.ones(nb_keypoints, np.float32),
            np.ones(nb_keypoints, np.float32),
        ],
        axis=-1,
    )
    colors = np.squeeze(cv2.cvtColor(colors[None, ...], cv2.COLOR_HSV2BGR))
    colors = (colors * 255).astype(np.uint8)
    for color, guesses in zip(colors, kps_by_kp):
        for kp in guesses:
            cv2.circle(image, tuple(map(int, kp[::-1])), 3, tuple(map(int, color)), -1)

    if err_rot is not None:
        cv2.putText(
            image,
            f"Rot Err: {np.degrees(err_rot):.2f}",
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )
    if err_pos is not None:
        cv2.putText(
            image,
            f"Pos Err: {err_pos:.2f}",
            (5, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
        )
    return image