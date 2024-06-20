# 图片暴力检测分类器

该仓库提供了使用预训练的ResNet模型将图像分类为暴力或正常类别的工具。以下是classify.py接口调用实例的说明。

## 模型准备

确保你有预训练的模型检查点文件。在初始化 `ViolenceClass` 时可以指定该文件的路径。

## 文件命名约定

数据集假设文件名包含标签，标签是文件名的第一部分，用下划线分隔。例如，`1_001.png` 表示该图像的标签为 `1`（暴力），`0_002.png` 表示该图像的标签为 `0`（正常）。
并且需要确保图像格式正确（JPEG, PNG），并且正确指定文件夹路径和模型检查点路径。

## 使用方法

### 1. 分类文件夹中的图像

你可以使用 `classify_folder` 方法对文件夹中的所有图像进行分类。

```python
from classify import ViolenceClass

# 模型检查点的路径
model_path = os.path.join(target_dir, 'train_logs', 'resnet18_pretrain_test', 'version_1', 'checkpoints', 'resnet18_pretrain_test-epoch=23-val_loss=0.08.ckpt')

# 初始化分类器
classifier = ViolenceClass(model_path)

# 包含图像的文件夹路径
folder_path = os.path.join(target_dir,'violence_224', 'test_val')

# 分类文件夹中的图像
folder_predictions, folder_probabilities = classifier.classify_folder(folder_path)
print(f'Folder image predictions: {folder_predictions}')
```

这将打印指定文件夹中每个图像的预测结果和置信度分数。图像将被分类为“暴力”或“正常”。

### 2. 测试模型准确率

你可以使用 `test_accuracy` 方法测试模型在数据集上的准确率。

```python
from classify import ViolenceClass

# 模型检查点的路径
model_path = os.path.join(target_dir, 'train_logs', 'resnet18_pretrain_test', 'version_1', 'checkpoints', 'resnet18_pretrain_test-epoch=23-val_loss=0.08.ckpt')

# 初始化分类器
classifier = ViolenceClass(model_path)

# 包含测试图像的文件夹路径
folder_path = os.path.join(target_dir,'violence_224', 'test_val')

# 测试模型在数据集上的准确率
classifier.test_accuracy(folder_path)
```

这将打印模型在指定文件夹中的测试集上的准确率。

## 示例

在 `classify.py` 脚本的主块中可以找到提供类的示例用法。根据你的设置修改 `model_path` 和 `folder_path` 变量，以分类图像并测试准确率。

```python
# 示例用法
if __name__ == "__main__":

    model_path = os.path.join(target_dir, 'train_logs', 'resnet18_pretrain_test', 'version_1', 'checkpoints', 'resnet18_pretrain_test-epoch=23-val_loss=0.08.ckpt')
    classifier = ViolenceClass(model_path)

    folder_path = os.path.join(target_dir,'violence_224', 'test_val')
    folder_predictions, folder_probabilities = classifier.classify_folder(folder_path)
    print(f'Folder image predictions: {folder_predictions}')

    classifier.test_accuracy(folder_path)
```

此示例初始化分类器，分类文件夹中的图像，并测试模型的准确率。你可以根据需要调整路径。