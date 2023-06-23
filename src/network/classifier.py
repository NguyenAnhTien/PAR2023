"""
@author: Tien Nguyen
@date  : 2023-05-15
"""
import random

from typing import Union
from typing import Optional

import numpy
import comet_ml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import constants
from network import Model
from dataset import DatasetHandler

class Classifier(pl.LightningModule):
    def __init__(
            self, configs
        ) -> None:
        super().__init__()
        self.configs = configs
        self.save_hyperparameters()
        self.set_seed(configs.seed)
        self.setup()
        self.define_model(configs)
        self.define_criterion()

    def forward(
            self, 
            image: torch.Tensor
        ) -> torch.Tensor:
        return self.model(image)

    @torch.no_grad()
    def predict(
            self,
            images: torch.Tensor
        ):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        outputs = self.model.predict(images)
        return outputs

    def training_step(
            self, 
            batch: dict, 
            batch_idx: int
        ) -> torch.Tensor:
        images = batch[constants.IMAGE]
        labels = batch[constants.LABEL]
        logits = self.model(images)
        loss = 0
        for item in zip(logits[0:2], labels[0:2]):
            loss += self.entropy_criterion(item[0], item[1])
        for item in zip(logits[2:], labels[2:]):
            pred = item[0].reshape(-1).type(torch.FloatTensor)
            pred = torch.sigmoid(pred)
            label = item[1].type(torch.FloatTensor)
            loss += self.binary_criterion(pred, label)
        preds = self.model.predict(images)
        _, mA = utils.cal_acc(preds, labels, device=self.configs.device)
        self.monitor(constants.TRAIN_ACC, mA.item())
        self.monitor(constants.TRAIN_LOSS, loss.detach())
        return loss

    def validation_step(
            self, 
            batch: dict, 
            batch_idx: int
        ) -> torch.Tensor:
        images = batch[constants.IMAGE]
        labels = batch[constants.LABEL]
        logits = self.model(images)
        loss = 0
        for item in zip(logits[0:2], labels[0:2]):
            loss += self.entropy_criterion(item[0], item[1])
        for item in zip(logits[2:], labels[2:]):
            pred = item[0].reshape(-1).type(torch.FloatTensor)
            pred = torch.sigmoid(pred)
            label = item[1].type(torch.FloatTensor)
            loss += self.binary_criterion(pred, label)
        preds = self.model.predict(images)
        _, mA = utils.cal_acc(preds, labels, device=self.configs.device)
        self.monitor(constants.VAL_ACC, mA.item())
        self.monitor(constants.VAL_LOSS, loss.detach())
        return loss

    def configure_optimizers(
            self
        ) -> torch.optim.SGD:
        return torch.optim.SGD(self.model.parameters(),\
                                            lr=self.configs.lr, momentum=0.9)
    
    def define_criterion(
            self
        ) -> None:
        self.entropy_criterion = torch.nn.CrossEntropyLoss()
        self.binary_criterion = torch.nn.BCELoss()

    def define_model(
            self,
            configs
        ) -> None:
        self.model = Model(configs=configs)

    def setup(
            self, 
            stage: Optional[str] = None
        ) -> None:
        train_samples_dir = utils.join_path((self.configs.preprocess_dir,\
                                                            constants.TRAIN))
        val_samples_dir = utils.join_path((self.configs.preprocess_dir,\
                                                                constants.VAL))
        self.train_data_handler = DatasetHandler(train_samples_dir)
        self.val_data_handler = DatasetHandler(val_samples_dir)

    def train_dataloader(
            self
        ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.train_data_handler,\
                                            batch_size=self.configs.batch_size,\
                                           shuffle=True,\
                                           num_workers=8, drop_last=True)
    
    def val_dataloader(
            self
        ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(dataset=self.val_data_handler,\
                                        batch_size=self.configs.batch_size,\
                                           shuffle=False,\
                                           num_workers=8, drop_last=True)

    def configure_callbacks(
            self
        ) -> list:
        MODEL_CKPT = 'models/model-{epoch:02d}-{val_loss:.2f}'
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',\
                                filename=MODEL_CKPT, mode='min', save_top_k=30)
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
        return [checkpoint_callback, lr_monitor]
    
    def monitor(
            self, 
            key: str, 
            value: float
        ) -> None:
        self.log(key, value, on_step=False, on_epoch=True,\
                            batch_size=self.configs.batch_size, sync_dist=True)

    def set_seed(
            self,
            seed: int
        ) -> None:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
