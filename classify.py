import model
from pytorch_lightning import Trainer
import torch
from torch.utils.data import DataLoader
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ViolenceClass:
    def __init__(self):
        self.VioCL = model.ViolenceClassifier.load_from_checkpoint("../NIS4307/train_logs/resnet18_pretrain_test/version_2/checkpoints/resnet18_pretrain_test-epoch=20-val_loss=0.03.ckpt")
        self.VioCL.to(device = 'cuda:0')
        self.trainer = Trainer(accelerator='gpu', devices=[0])

    def classify(self, imgs : torch.Tensor):
        batchsize = imgs.size(0)
        preds = [-1]*batchsize
        pred_dataloader = DataLoader(imgs, batch_size=batchsize)
        with torch.no_grad():
            prediction_scores = self.trainer.predict(self.VioCL,dataloaders=pred_dataloader,return_predictions=True)
        prediction_scores = prediction_scores[0]
        i = 0
        for cls_score in prediction_scores:
            if cls_score[0] > cls_score[1]:
                preds[i] = 0
            else:
                preds[i] = 1
            i += 1
        return preds