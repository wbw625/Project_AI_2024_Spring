
import os

# 获取当前脚本文件的路径
script_dir = os.path.dirname(__file__)

# 相对路径到目标目录
target_dir = os.path.join(script_dir, '10-其他支持文件和目录')


import torch
import os
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
        self.VioCL.eval()  # 设置模型为评估模式

    def classify(self, imgs: torch.Tensor):
        imgs = imgs.to(self.device)
        with torch.no_grad():
            outputs = self.VioCL(imgs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities[:, 1] > 0.5).long()
        return predictions.cpu().tolist(), probabilities.cpu().tolist()

    def classify_folder(self, folder_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        imgs = []
        img_names = []
        print(f"Reading images from folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                try:
                    image = Image.open(img_path).convert('RGB')
                    imgs.append(transform(image))
                    img_names.append(filename)
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")

        if not imgs:
            print(f"No images found in folder: {folder_path}")
            return []

        imgs_tensor = torch.stack(imgs)
        predictions, probabilities = self.classify(imgs_tensor)


        return predictions, probabilities

    def test_accuracy(self, folder_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = ViolenceDataset(folder_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        correct = 0
        total = 0
        for imgs, labels in dataloader:
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            preds, _ = self.classify(imgs)
            correct += (torch.tensor(preds).to(self.device) == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
        return accuracy

# 示例用法
if __name__ == "__main__":

    #model_path = os.path.join(target_dir, 'train_logs', 'resnet18_pretrain_test', 'version_2', 'checkpoints', 'resnet18_pretrain_test-epoch=15-val_loss=0.08.ckpt')
    model_path = os.path.join(target_dir, 'train_logs', 'resnet18_pretrain_test', 'version_1', 'checkpoints', 'resnet18_pretrain_test-epoch=23-val_loss=0.08.ckpt')

    classifier = ViolenceClass(model_path)

    folder_path = os.path.join(target_dir,'violence_224', 'test_val')
    #folder_path = os.path.join(target_dir,'violence_224', 'test_set2')
    #folder_path = os.path.join(target_dir,'violence_224', 'test_set3')
    folder_predictions, folder_probabilities = classifier.classify_folder(folder_path)
    print(f'Folder image predictions: {folder_predictions}')

    classifier.test_accuracy(folder_path)
