from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule


if __name__ == '__main__':
    gpu_id = [1]
    lr = 3e-4
    batch_size = 128
    log_name = "resnet18_pretrain_test"
    print("{} gpu: {}, batch size: {}, lr: {}".format(log_name, gpu_id, batch_size, lr))

    data_module = CustomDataModule(batch_size=batch_size)
    # 设置模型检查点，用于保存最佳模型
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=log_name + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    logger = TensorBoardLogger("train_logs", name=log_name)

    # 实例化训练器，移除GPU相关设置
    trainer = Trainer(
        max_epochs=20,
        accelerator='cpu',  # 使用CPU进行训练
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # 实例化模型
    #model = ViolenceClassifier(learning_rate=lr)

    # 实例化模型并加载预训练权重
    model_path = 'train_logs/resnet18_pretrain_continued/version_0/checkpoints/resnet18_pretrain_continued-epoch=10-val_loss=0.04.ckpt'  # 替换为之前训练模型的检查点路径
    model = ViolenceClassifier.load_from_checkpoint(model_path, learning_rate=lr)

    # 开始训练
    trainer.fit(model, data_module)

    
