import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ViT import ViT
# 训练参数
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 3e-4

# 获取数据
train_dataset = datasets.CIFAR10(root='./data', transform=ToTensor(), train=True, download=True)
test_dataset = datasets.CIFAR10(root='./data', transform=ToTensor(), train=False, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 实例化模型
model = ViT(image_size=32, patch_size=16, hidden_size=512, n_layers=6, n_heads=8, num_classes=10).to(device)

# 定义优化器和损失函数 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练
for epoch in range(EPOCHS):
    train_loss = 0
    for batch in train_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计loss
        train_loss += loss.item()
    print(f"Epoch {epoch+1} - Train loss: {train_loss/len(train_loader)}")

# 测试
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_loader:
        images = batch[0].to(device)
        labels = batch[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('Test Accuracy: {} %'.format(100 * correct / total)) 
# 这个训练代码实现了:
# 1. 获取CIFAR10数据集
# 2. 实例化ViT模型和优化器等
# 3. 迭代EPOCHS轮epoch
# 4. 每个epoch训练一次全部训练集,统计loss,优化模型
# 5. 每个epoch结束测试在测试集上的精度
# 6. 完成训练后打印最终测试精度
# 使用这个训练代码,可以对ViT模型进行训练和评估。由于计算资源有限,这里只训练了10个epoch,精度可能不太高,但可以展示训练流程。可以根据自己的实际情况修改超参数,训练更多epoch,获得更高精度。
# 希望这个实现可以提供一定帮助,有任何问题和改进建议都请提出,我很乐意讨论和学习。