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
        num_ftrs = self.model.num_features
        self.fc = nn.Linear(1000, num_ftrs)
        self.fc1 = nn.Linear(num_ftrs, 12)
        self.fc2 = nn.Linear(num_ftrs, 12)
        self.fc3 = nn.Linear(num_ftrs, 1)
        self.fc4 = nn.Linear(num_ftrs, 1)
        self.fc5 = nn.Linear(num_ftrs, 1)
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
        outputs = self.model(images)
        
        outputs = outputs.view(outputs.size(0), -1)
        outputs = self.fc(outputs)
        color_top = self.relu(self.fc1(outputs))
        color_bottom = self.relu(self.fc2(outputs))
        
        gen = self.relu(self.fc3(outputs))
        bag = self.relu(self.fc4(outputs))
        hat = self.relu(self.fc5(outputs))
        
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
