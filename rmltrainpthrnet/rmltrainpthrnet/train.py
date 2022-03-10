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
import torch.nn as nn
import torch.optim as optim
## define trainer here

## TODO: optimizer options, model options, validation from direction, set step pose error, comet logging
class HRNET(pl.LightningModule):

    def __init__(self, hp, data_dir, artifact_dir, keypoints_3d):
        super().__init__()
        self.hp = hp
        self.nb_keypoints = hp["keypoints"]
        self.data_dir = data_dir
        self.artifact_dir = artifact_dir
        self.keypoints_3d = keypoints_3d[: self.nb_keypoints]
        self.model = get_pose_net(self.hp, True)
        self.loss = MultiHeatMapLoss()
    
    def forward(self, images):  
        return self.model(images)
    
    def train_dataloader(self):
        ds = HeatMapDataset(self.hp, self.data_dir, self.nb_keypoints, "train")
        loader = torch.utils.data.DataLoader(ds, batch_size=self.hp["batch_size"])
        return loader

    def test_dataloader(self):
        ds = HeatMapDataset(self.hp, self.data_dir, self.nb_keypoints, "test")
        loader = torch.utils.data.DataLoader(ds, batch_size=1)
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hp["learning_rate"]) ##TODO: more optimizer options
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1) ##TODO: better LR schedule
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        images, heatmaps = batch
        heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))
        y_hat = self.model(images)
        loss = self.loss(y_hat, heatmaps)
        self.log("train_loss", loss)
        return loss

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
        self.nb_keypoints = nb_keypoints
        self.loss_factor = loss_factor
        self.heatmap_losses =\
            nn.ModuleList(
                [
                    HeatmapLoss()
                    for _ in self.nb_keypoints
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
