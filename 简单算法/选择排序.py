def selected_sort(arry):

 # 已经排好序的末尾索引  124 3 65 选取最小的放在已经排好序的末尾就是从小到大排
    for i in range(0,len(arry)):
        min = i
        for j in range(i + 1,len(arry)):

            if arry[j] < arry[min]:
                min = j

        if min != i:  # 如果最小值不是当前值
            arry[i], arry[min] = arry[min], arry[i]

    return arry

# 测试
if __name__ == "__main__":
    arry = [154, 3, 62, 1, 5, 6, 7, 8, 9]
    print("排序前：", arry)
    sorted_arry = selected_sort(arry)
    print("排序后：", sorted_arry)