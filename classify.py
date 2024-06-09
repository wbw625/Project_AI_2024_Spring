import torch
import os
import model
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""

class ViolenceClass:
    def __init__(self, model_path):
        self.VioCL = model.ViolenceClassifier.load_from_checkpoint(model_path)
        self.VioCL.to(device=device)
        self.trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=[0] if torch.cuda.is_available() else 1)

    def classify(self, imgs: torch.Tensor):
        batchsize = imgs.size(0)
        preds = [-1] * batchsize
        pred_dataloader = DataLoader(imgs, batch_size=batchsize)
        with torch.no_grad():
            prediction_scores = self.trainer.predict(self.VioCL, dataloaders=pred_dataloader, return_predictions=True)
        prediction_scores = prediction_scores[0]
        i = 0
        for cls_score in prediction_scores:
            if cls_score[0] > cls_score[1]:
                preds[i] = 0
            else:
                preds[i] = 1
            i += 1
        return preds

    def classify_folder(self, folder_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        imgs = []
        print(f"Reading images from folder: {folder_path}")
        for filename in os.listdir(folder_path):
            # print(f"Found file: {filename}")
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                try:
                    image = Image.open(img_path)
                    image = transform(image)
                    imgs.append(image)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

        if len(imgs) == 0:
            print(f"No images found in folder: {folder_path}")
            return []

        imgs = torch.stack(imgs)
        return self.classify(imgs)

# 示例用法
if __name__ == "__main__":
    # 假设模型文件路径
    model_path = 'train_logs/resnet18_pretrain_test/version_1/checkpoints/resnet18_pretrain_test-epoch=35-val_loss=0.00.ckpt'

    # 初始化分类器
    classifier = ViolenceClass(model_path)

    # 进行文件夹图像预测的示例
    folder_path = 'violence_224/val'
    folder_predictions = classifier.classify_folder(folder_path)
    print(f'Folder image predictions: {folder_predictions}')
