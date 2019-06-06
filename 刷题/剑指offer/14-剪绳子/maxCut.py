"""
把长度为n的绳子剪成m段，m,n均为整数，问m小段绳子长度之积最大为多少？
"""

# -*- coding:utf-8 -*-
class Solution:

    def __init__(self):
        self.product = {0: 0, 1: 1, 2: 2, 3: 3}  # 长度为key的绳子的最优解为value

    def maxCut1(self, length):  # 动态规划解答
        if length < 0:
            return 0
        elif length < 4:
            return self.product[length]
        for i in range(4, length + 1):
            maxProduct = 0
            for j in range(1, i//2 + 1):
                tmpProduct = self.product[j] * self.product[i - j]
                if maxProduct < tmpProduct:
                    maxProduct = tmpProduct
            self.product[i] = maxProduct
        return self.product[length]

    def maxCut2(self, length):  # 贪心算法求解
        if length < 0:
            return 0
        elif length < 5:
            return length
        else:
        #if (length - 3) > 0:  # 当 length >= 5时，尽量裁剪成长度为3的小绳子就完事了
            return 3 * self.maxCut2(length - 3)



if __name__ == "__main__":
    f = Solution()
    length = 8
    print("动态规划求解：")
    print(f.maxCut1(length))
    print(f.product)
    print("贪心算法求解：")
    print(f.maxCut2(length))