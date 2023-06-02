"""
@author : Khanh Tran
@date   : 2023-05-08
@update : Tien Nguyen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import utils

class Model(nn.Module):
    def __init__(
            self, 
            configs
        ) -> None:
        super(Model, self).__init__()
        self.configs = configs
        resnet = models.resnet152(weights=True)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, len(self.configs.num_classes))
        self.fc = nn.Linear(num_ftrs * 2 * 2, num_ftrs)
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.fc1 = nn.Linear(num_ftrs, 12)
        self.fc2 = nn.Linear(num_ftrs, 12)
        self.fc3 = nn.Linear(num_ftrs, 1)
        self.fc4 = nn.Linear(num_ftrs, 1)
        self.fc5 = nn.Linear(num_ftrs, 1)
        self.relu = nn.ReLU()

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
