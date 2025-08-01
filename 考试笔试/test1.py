#1、请使用Python的NumPy库创建两个长度为100的随机一维数组A和B，
# 数组中的元素为0到9之间的整数。然后从数组A中移除与数组B重复的项，并输出最终的数组A。
import numpy as np

# 设置随机种子以便结果可复现
np.random.seed(42)

# 创建两个长度为100的随机一维数组A和B，元素为0到9之间的整数
A = np.random.randint(0, 10, 100)
B = np.random.randint(0, 10, 100)



# 从数组A中移除与数组B重复的项
# 使用numpy的setdiff1d函数，返回A中不在B中的元素
A_filtered = np.setdiff1d(A, B)

