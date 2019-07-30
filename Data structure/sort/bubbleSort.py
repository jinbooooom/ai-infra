# -*- coding:utf-8 -*-

class Sort:
    def bubbleSort(self, l):
        # 需要 len(l)-1 次遍历
        for i in range(len(l) - 1):
            # 第 i 次遍历，说明已经使 i 个大的数沉底，那么只需要对列表l[:len(l)-i]部分相邻的元素比较大小
            # 且只需要比较 len(l)-i-1 次
            print(l)
            for j in range(len(l) - i - 1):
                if l[j] > l[j + 1]:
                    l[j], l[j + 1] = l[j + 1], l[j]
        return l

if __name__ == "__main__":
    f = Sort()
    l = [9, 8, 5, 6, 3]
    print(f.bubbleSort(l))
