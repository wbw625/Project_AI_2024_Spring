import torch
import os
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from model import ViolenceClassifier

class ViolenceClass:
    def __init__(self, model_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else ""
        self.VioCL = ViolenceClassifier.load_from_checkpoint(model_path).to(device=self.device)
        self.trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices=[0] if torch.cuda.is_available() else 1)

    def classify(self, imgs: torch.Tensor):
        pred_dataloader = DataLoader(imgs, batch_size=imgs.size(0))
        with torch.no_grad():
            prediction_scores = self.trainer.predict(self.VioCL, dataloaders=pred_dataloader, return_predictions=True)
        return [1 if score[1] > score[0] else 0 for score in prediction_scores[0]]

    def classify_folder(self, folder_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        imgs = []
        print(f"Reading images from folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                try:
                    image = Image.open(img_path).convert('RGB')
                    imgs.append(transform(image))
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

        if not imgs:
            print(f"No images found in folder: {folder_path}")
            return []

        return self.classify(torch.stack(imgs))

# Example usage
if __name__ == "__main__":
    model_path = 'train_logs/resnet18_pretrain_test/version_1/checkpoints/resnet18_pretrain_test-epoch=35-val_loss=0.00.ckpt'
    classifier = ViolenceClass(model_path)
    folder_path = 'violence_224/val'
    folder_predictions = classifier.classify_folder(folder_path)
    print(f'Folder image predictions: {folder_predictions}')
