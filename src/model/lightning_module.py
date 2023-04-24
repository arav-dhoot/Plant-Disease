import timm
import wandb
import torch.optim
import torchmetrics
import torch.nn as nn
from timm.data import resolve_data_config
from timm.loss import SoftTargetCrossEntropy
from pytorch_lightning.core.module import LightningModule
from timm.data.transforms_factory import create_transform

class LitModel(LightningModule):
    def __init__(self, 
                num_classes, 
                model, 
                learning_rate, 
                weight_decay, 
                dropout_rate=0.5, 
                fine_tune=False, 
                mixup_func=None):
        
        super().__init__()
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.mixup_func = mixup_func
        self.dropout = dropout_rate
        self.accuracy = torchmetrics.classification.MulticlassAccuracy(self.num_classes)
        self.model = model
        self.criterion = SoftTargetCrossEntropy() if (self.mixup_func is not None) else nn.CrossEntropyLoss()

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
        
        config = {
            'model': self.model,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'droupout': self.dropout
        }
        
        self.wandb_run = wandb.init(
            project='plant_pathology',
            id = f'{self.num_classes}',
            config=config
        )

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
            acc = self.accuracy(probabilities, y)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.wandb_run.log({'training loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        representations = self.feature_extractor(x)
        logits = self.classifier(representations)
        probabilities = torch.softmax(logits, dim=1)
        y = torch.Tensor(y)
        loss = self.criterion(probabilities, y)
        # accuracy = torchmetrics.functional.accuracy(probabilities, y, task='multiclass')
        acc = self.accuracy(probabilities, y)
        self.log("accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.int64)
        import pdb; pdb.set_trace()
        representations = self.feature_extractor(x.float())
        logits = self.classifier(representations)
        probabilities = torch.softmax(logits, dim=1)
        y = torch.Tensor(y)
        loss = self.criterion(probabilities, y)
        acc = self.accuracy(probabilities, y)
        self.log("accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.wandb_run.log({'test loss': loss})
        self.wandb_run.log({'accuracy': acc})
        return loss

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))
                         
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)