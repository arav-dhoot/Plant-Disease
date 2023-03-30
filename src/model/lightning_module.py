import pytorch_lightning as pl
import torchmetrics
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from timm.loss import SoftTargetCrossEntropy
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np

class LitModel(LightningModule):
    def __init__(self, num_classes, model, learning_rate, weight_decay, dropout_rate=0.5, fine_tune=False, mixup_func=None):
        super().__init__()
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.mixup_func = mixup_func
        self.model = model
        if model == 'ViT':
            self.feature_extractor = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0)
            config = resolve_data_config({}, model=self.feature_extractor)
            transform = create_transform(**config)
            self.classifier = nn.Linear(768, self.num_classes)

        elif model == 'ResNet':
            self.feature_extractor = timm.create_model('resnet50', pretrained=True, num_classes=0)
            config = resolve_data_config({}, model=self.feature_extractor)
            transform = create_transform(**config)
            self.classifier = nn.Linear(2048, self.num_classes)

        if (not fine_tune):
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        if (self.mixup_func is not None):
            self.criterion = SoftTargetCrossEntropy()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch    
        representations = self.feature_extractor(x)
        logits = self.classifier(representations)
        probabilities = torch.softmax(logits, dim=1)
        if(self.mixup_func is not None):
            x, y = self.mixup_func(x, y)
        y = torch.Tensor(y)
        loss = self.criterion(probabilities, y)
        if self.mixup_func is None:
            accuracy = torchmetrics.functional.accuracy(probabilities, y)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        representations = self.feature_extractor(x)
        logits = self.classifier(representations)
        probabilities = torch.softmax(logits, dim=1)
        y = torch.Tensor(y)
        loss = self.criterion(probabilities, y)
        accuracy = torchmetrics.functional.accuracy(probabilities, y)
        self.log("accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        representations = self.feature_extractor(x)
        logits = self.classifier(representations)
        probabilities = torch.softmax(logits, dim=1)
        y = torch.Tensor(y)
        loss = self.criterion(probabilities, y)
        accuracy = torchmetrics.functional.accuracy(probabilities, y)
        self.log("accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))
                         
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)