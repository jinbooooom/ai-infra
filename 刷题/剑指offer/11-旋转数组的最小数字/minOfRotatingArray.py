"""
把数组最开始的若干个元素搬到数组尾部称为数组的旋转，
对排序的递增数组的一个旋转，找到该旋转数组的最小元素
"""

# -*- coding:utf-8 -*-
class Solution:
    def minOfRotatingArray(self, L):
        # write code here
        if not len(L):
            return
        if L[0] < L[-1]:  # 把数组开头的零个元素搬到数组尾部，即数组未变，为递增排序的数组。
            return L[0]
        P1 = 0
        P2 = len(L) - 1
        while L[P2] <= L[P1]:
            if P2 - P1 == 1:
                mid = P2
                break
            mid = (P1 + P2)//2
            if L[P1] == L[mid] and L[mid] == L[P2]:
                result = L[P1]
                for i in L[P1+1:P2+1]:
                    if result > i:
                        result = i
                return result
            if L[mid] >= L[P1]:
                P1 = mid
            elif L[mid] <= L[P2]:
                P2 = mid
        return L[mid]


if __name__ == "__main__":
    L = [[3,4,5,1,2],
         [2,2,2,2,2],
         [2,3,1,2,2],
         [1,2,3,4,5],
         [1,0,1,1,1],
         [1,1,1,0,1],
         ]
    f = Solution()
    for l in L:
        print(l)
        print('min:', f.minOfRotatingArray(l))
