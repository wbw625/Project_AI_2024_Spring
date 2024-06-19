import torch
import os
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from model import ViolenceClassifier

class ViolenceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for filename in os.listdir(self.folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                label = int(filename.split('_')[0])  # 从文件名提取标签
                self.labels.append(label)
                img_path = os.path.join(self.folder_path, filename)
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.images.append(image)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class ViolenceClass:
    def __init__(self, model_path):
        self.device = 'cpu'  # 使用CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用GPU
        self.VioCL = ViolenceClassifier.load_from_checkpoint(model_path).to(device=self.device)
        self.trainer = Trainer(accelerator='cpu', devices=1)  # 使用CPU

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

    def test_accuracy(self, folder_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = ViolenceDataset(folder_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        correct = 0
        total = 0
        for imgs, labels in dataloader:
            imgs = imgs.to(self.device)  
            labels = labels.to(self.device)  
            preds = self.classify(imgs)
            correct += (torch.tensor(preds).to(self.device) == labels).sum().item()  # 确保预测结果在正确的设备上
            total += labels.size(0)

        accuracy = correct / total
        print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
        return accuracy

# 示例用法
if __name__ == "__main__":
    model_path = 'train_logs\\resnet18_pretrain_test\\version_8\\checkpoints\\resnet18_pretrain_test-epoch=23-val_loss=0.08.ckpt'
    classifier = ViolenceClass(model_path)
    # folder_path = 'violence_224/val'
    # folder_path = 'train set 2'
    folder_path = 'train set 3'
    # folder_path = 'violence_224/test'
    folder_predictions = classifier.classify_folder(folder_path)
    print(f'Folder image predictions: {folder_predictions}')

    classifier.test_accuracy(folder_path)