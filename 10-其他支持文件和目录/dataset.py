from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from pytorch_lightning import LightningDataModule

class CustomDataset(Dataset):
    def __init__(self, split):
        assert split in ["train", "val", "test"]
        data_root = "E:\\2024-1\\Intro_to_AI\\ai_img_srt\\violence_224"
        self.data = [os.path.join(data_root, split, i) for i in os.listdir(os.path.join(data_root, split))]
        
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # 将图像转换为Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # 将图像转换为Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        x = Image.open(img_path).convert('RGB')
        
        # 获取标签值，假设文件名以“0_”或“1_”开头
        filename = os.path.basename(img_path)
        try:
            y = int(filename.split("_")[0])
        except ValueError:
            print(f"无法从文件名中提取标签：{filename}")
            raise
        if y not in [0, 1]:
            print(f"错误的标签值：{y}，文件名：{filename}")
            raise ValueError(f"错误的标签值：{y}，文件名：{filename}")
        
        x = self.transforms(x)
        return x, y

class CustomDataModule(LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        self.train_dataset = CustomDataset("train")
        self.val_dataset = CustomDataset("val")
        self.test_dataset = CustomDataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
