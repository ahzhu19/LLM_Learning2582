# -*- coding: utf-8 -*-
# @Time    : 2025/7/29 16:36
# @Author  : 风华正茂心
# @FileName: rank.py
# @Software: PyCharm

# @Description: 排序算法实现
# 选择排序
# range(i, len) 生成一个从 i 到 len-1 的整数序列

def slection_sort(arr):
    len = len(arr)
    for i in range(len):
        min_index = i

        for j in range(i+1, len):
            if arr[j] < arr[min_index]:
                min_index = j
                # 交换位置
                if min_index != i:
                    arr[i], arr[min_index] = arr[min_index], arr[i]
            j += 1
