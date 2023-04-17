# 导入所需的模块
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os 
os.getcwd()
# 读取.npz格式的数据文件
data = np.load('/home/mulati/Murat/Group/脉冲图像重构代码/SpikeFormer-main/Dataset/train/000_part_1_id_0.npz')

# 获取文件中的所有数组名称
names = data.files

# 打印数组名称和数组值
for name in names:
    print(name)
    print(data[name])

# 选择一个数组进行可视化，例如第一个数组
array = data[names[0]]
array.shape

names


i = 0
k = 7
array = data[names[i]]
array.shape
# 绘制图像
plt.imshow(array[k], cmap='gray')
# 设置标题和颜色条
plt.title('Image of ' + names[i])
plt.colorbar()
# 显示图像
plt.show()



#版本1
# 选择两个数组进行可视化，分别是'spSeq'和'gt'
spSeq = data[names[0]] # 脉冲图片
gt = data[names[1]] # 真实图片

# 判断两个数组的维度，如果是三维，可以使用plt.subplot函数绘制多个子图
if spSeq.ndim == 3 and gt.ndim == 3:
    # 获取第一个维度的大小，即图片的数量
    n = spSeq.shape[0]
    # 创建一个大图，设置大小和标题
    plt.figure(figsize=(10, 10))
    plt.suptitle('Pulse images and ground truth images')
    # 循环遍历每个图片
    for i in range(n):
        # 绘制脉冲图片，放在左边的列
        plt.subplot(n, 2, i * 2 + 1)
        plt.imshow(spSeq[i], cmap='gray')
        # 设置子标题和颜色条
        plt.title('Pulse image ' + str(i))
        plt.colorbar()
        # 绘制真实图片，放在右边的列
        plt.subplot(n, 2, i * 2 + 2)
        plt.imshow(gt[i], cmap='gray')
        # 设置子标题和颜色条
        plt.title('Ground truth image ' + str(i))
        plt.colorbar()
    # 显示大图
    plt.show()




#版本2
# 判断两个数组的维度，如果是三维，可以使用plt.subplot函数绘制多个子图
if spSeq.ndim == 3 and gt.ndim == 3:
    # 获取第一个维度的大小，即图片的数量
    n = spSeq.shape[0]
    # 创建一个大图，设置大小和标题
    plt.figure(figsize=(15, 15 * n)) # 增加图像大小
    plt.suptitle('Pulse images and ground truth images')
    # 循环遍历每个图片
    for i in range(n):
        # 绘制脉冲图片，放在左边的列
        plt.subplot(n, 2, i * 2 + 1)
        plt.imshow(spSeq[i], cmap='gray')
        # 设置子标题和颜色条
        plt.title('Pulse image ' + str(i))
        plt.colorbar()
        # 绘制真实图片，放在右边的列
        plt.subplot(n, 2, i * 2 + 2)
        plt.imshow(gt[i], cmap='gray')
        # 设置子标题和颜色条
        plt.title('Ground truth image ' + str(i))
        plt.colorbar()
    # 调整子图之间的间距，避免重叠
    plt.tight_layout()
    # 显示大图
    plt.show()