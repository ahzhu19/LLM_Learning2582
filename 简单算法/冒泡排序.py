

def switch(arr,i,j):
    arr[i],arr[j] = arr[j],arr[i]
def bubble(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0,n-i-1):
            if arr[j] > arr[j+1]:
                switch(arr,j,j+1)

    return arr
# 测试
if __name__ == "__main__":
    arr = [154, 3, 62, 1, 5, 6, 7, 8, 9]
    print("排序前：", arr)
    sorted_arr = bubble(arr)
    print("排序后：", sorted_arr)