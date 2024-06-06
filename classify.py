import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ViolenceClass:
    def __init__(self, model_path, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_path):
        model = models.resnet18(pretrained=False, num_classes=2)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model

    def classify(self, imgs: torch.Tensor) -> list:
        imgs = imgs.to(self.device)
        with torch.no_grad():
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().tolist()

# 示例代码，展示如何使用这个类
if __name__ == "__main__":
    model_path = "/path/to/saved_model/resnet18_pretrain_test-epoch=03-val_loss=0.50.ckpt"  # 替换为实际的模型路径
    classifier = ViolenceClass(model_path)

    # 加载并预处理测试图像
    test_image_paths = ["path/to/image1.jpg", "path/to/image2.jpg"]  # 替换为实际的图像路径
    test_images = [Image.open(img_path) for img_path in test_image_paths]
    test_tensors = torch.stack([classifier.transform(img) for img in test_images])

    # 进行分类
    predictions = classifier.classify(test_tensors)
    print(predictions)  # 输出预测结果

