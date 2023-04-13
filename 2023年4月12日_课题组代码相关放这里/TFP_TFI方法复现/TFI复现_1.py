
# tfi方法的基本思想是利用脉冲间隔（ISI）的信息来恢复图像的纹理，即认为ISI越小，对应的像素点越亮，反之越暗。具体的步骤如下：

# 首先，导入所需的库和函数，如numpy, scipy.io, matplotlib等
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# 然后，读取data里的.mat文件，提取出脉冲数据和时间戳数据。假设文件名为data.mat，脉冲数据为spike，时间戳数据为time
data = sio.loadmat('./data/train/data.mat')
spike = data['spike']
time = data['time']

# data = np.load('./data/test/220_part_1_id_0.npz')
# print(data.files)
# spike = data['spike']
# time = data['time']

# # 获取文件中的所有数组名称
# names = data.files
# # 打印数组名称和数组值
# for name in names:
#     print(name)
#     print(data[name])
# # 选择一个数组进行可视化，例如第一个数组
# array = data[names[0]]
# array.shape
# names
# i = 0
# k = 7
# array = data[names[i]]
# array.shape
# # 绘制图像
# plt.imshow(array[k], cmap='gray')
# # 设置标题和颜色条
# plt.title('Image of ' + names[i])
# plt.colorbar()
# # 显示图像
# plt.show()

# 接下来，对脉冲数据进行预处理，去除噪声和异常值，以及根据时间戳对脉冲进行排序。这里我们假设噪声和异常值的判断标准是脉冲数小于等于1或者大于等于10的像素点，可以用numpy的where函数来找出这些像素点，并将其置为0。然后，我们可以用numpy的argsort函数来根据时间戳对脉冲进行排序。
# 去除噪声和异常值
noise = np.where((spike <= 1) | (spike >= 10))
spike[noise] = 0

# 根据时间戳排序
index = np.argsort(time, axis=1)
spike_sorted = np.take_along_axis(spike, index, axis=1)
time_sorted = np.take_along_axis(time, index, axis=1)
# 然后，计算每个像素点在一定时间窗口内的平均ISI。这里我们假设时间窗口为100毫秒，即只考虑每个像素点在最近100毫秒内发生的脉冲。我们可以用numpy的diff函数来计算相邻两个脉冲之间的时间差，即ISI。然后，我们可以用numpy的cumsum函数来计算每个像素点在每个时刻之前发生的所有脉冲的累积时间差，即累积ISI。接着，我们可以用numpy的searchsorted函数来找出每个像素点在最近100毫秒内发生的第一个脉冲的位置，并用numpy的take_along_axis函数来提取出这些位置对应的累积ISI。最后，我们可以用numpy的mean函数来计算每个像素点在最近100毫秒内发生的所有脉冲的平均ISI。
# 计算相邻两个脉冲之间的时间差
isi = np.diff(time_sorted, axis=1)

# 计算每个像素点在每个时刻之前发生的所有脉冲的累积时间差
cum_isi = np.cumsum(isi, axis=1)

# 找出每个像素点在最近100毫秒内发生的第一个脉冲的位置
window = 100  # 时间窗口为100毫秒
start = np.searchsorted(cum_isi, time_sorted - window, side='right')

# 提取出这些位置对应的累积ISI
cum_isi_start = np.take_along_axis(cum_isi, start[:, None], axis=1)


avg_isi = (cum_isi - cum_isi_start) / (time_sorted -
                                       time_sorted[:, 0][:, None])  # 计算每个像素点在最近100毫秒内发生的所有脉冲的平均ISI

# 接下来，将每个像素点的平均ISI映射到0-255的范围内，并将其转换为图像格式。我们可以用numpy的clip函数来将平均ISI限制在一个最大值和最小值之间，然后用numpy的interpolate函数来将其线性插值到0-255之间。然后，我们可以用matplotlib的imshow函数来显示重构的图像。
# 将平均ISI映射到0-255的范围内
max_isi = 100  # 最大ISI
min_isi = 10  # 最小ISI
isi_clipped = np.clip(avg_isi, min_isi, max_isi)
isi_scaled = np.interp(isi_clipped, (min_isi, max_isi), (0, 255))

# 转换为图像格式
image = isi_scaled.astype(np.uint8)

# 显示重构的图像
plt.imshow(image, cmap='gray')
plt.show()

