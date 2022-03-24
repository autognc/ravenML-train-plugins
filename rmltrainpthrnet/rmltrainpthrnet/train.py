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
from rmltrainpthrnet.utils.heatmaps import keypoints_from_output
import rmltrainpthrnet.utils.pose as pose_utils
import torch.nn as nn
import torch.optim as optim
## define trainer here

## TODO: optimizer options, model options, validation from directory, val step pose error, comet logging
class HRNET(pl.LightningModule):

    def __init__(self, hp, data_dir, artifact_dir, keypoints_3d):
        super().__init__()
        self.hp = hp
        self.nb_keypoints = hp["keypoints"]
        self.data_dir = data_dir
        self.artifact_dir = artifact_dir
        self.keypoints_3d = keypoints_3d[: self.nb_keypoints]
        self.focal_length = self.hp["pnp_focal_length"]
        self.model = get_pose_net(self.hp, True)
        self.loss = MultiHeatMapLoss(self.nb_keypoints)
    
    def forward(self, images):  
        return self.model(images)
    
    def train_dataloader(self):
        ds = HeatMapDataset(self.hp, self.data_dir, self.nb_keypoints, "train").dataset
        loader = torch.utils.data.DataLoader(ds, num_workers=self.hp.dataset.num_workers, batch_size=self.hp["batch_size"])
        #loader = torch.utils.data.DataLoader(ds, batch_size=self.hp["batch_size"])
        return loader

    def val_dataloader(self):
        ds = HeatMapDataset(self.hp, self.data_dir, self.nb_keypoints, "test").dataset
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp["learning_rate"]) ##TODO: more optimizer options
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1) ##TODO: better LR schedule
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
        
        keypoints_batch = keypoints_from_output(heatmap_pred, self.hp.dataset.input_size, 10).cpu().numpy()
        keypoints_batch.transpose(0,2,1,3)
        pose_errs = []
        trans_errs = []
        for i, kps in enumerate(keypoints_batch):
            imdims_single = imdims[i].cpu().numpy()
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
        self.log("validation_pose_err", np.degrees(np.mean(pose_errs)), prog_bar=True)
        self.log("validation_translation_err", np.mean(trans_errs), prog_bar=True)    
        return total_loss

class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
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
