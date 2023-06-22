"""
@author : Khanh Tran
@date   : 2023-05-08
@update : Tien Nguyen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import timm

import utils

class Model(nn.Module):
    def __init__(
            self, 
            configs
        ) -> None:
        super(Model, self).__init__()
        self.configs = configs
        self.model = self.define_model(configs.model_name)
        self.fc1 = nn.Linear(self.model.in_features, 12)
        self.fc2 = nn.Linear(self.model.in_features, 12)
        self.fc3 = nn.Linear(self.model.in_features, 1)
        self.fc4 = nn.Linear(self.model.in_features, 1)
        self.fc5 = nn.Linear(self.model.in_features, 1)
        self.relu = nn.ReLU()

    def define_model(
            self,
            model_name: str = 'swin_tiny_patch4_window7_224',
        ) -> nn.Module:
        model = timm.create_model(model_name, pretrained=True)
        return model

    def forward(
            self, 
            images: torch.Tensor
        ) -> tuple:
        images = self.resnet(images)
        
        images = images.view(images.size(0), -1)
        
        images = self.fc(images)
        color_top = self.relu(self.fc1(images))
        color_bottom = self.relu(self.fc2(images))
        
        gen = self.relu(self.fc3(images))
        bag = self.relu(self.fc4(images))
        hat = self.relu(self.fc5(images))
        
        return color_top, color_bottom, gen, bag, hat

    @torch.no_grad()
    def predict(
            self, 
            images: torch.Tensor
        ):
        logits = self(images)
        logits = utils.concat_tensors(logits, device=self.configs.device)
        logits = torch.transpose(logits, 0, 1)
        preds = torch.argmax(logits, dim=2)
        return preds
