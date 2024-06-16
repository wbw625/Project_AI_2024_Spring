import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torchvision import models
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule

class ViolenceClassifier(LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        self.learning_rate = learning_rate
        # 定义权重，假设类别0的权重为2，类别1的权重为1
        weights = torch.tensor([2, 1], dtype=torch.float)
        # 定义新的损失函数
        self.loss_fn = BCEWithLogitsLoss(pos_weight=weights)
        self.accuracy = Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # 定义优化器
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # 将目标转换为one-hot编码
        y_onehot = F.one_hot(y, num_classes=2)
        loss = self.loss_fn(logits, y_onehot.float())
        return loss