"""我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？"""

"""
最后一次覆盖是竖着覆盖则前面的2*(n-1)矩形有f(n-1)种方法，
最后一次覆盖是横着覆盖则前面的2*(n-2)矩形有f(n-2)种方法"""

# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, n):
        # write code here
        if n <= 0:
            return 0
        if n == 1 or n == 2:
            return n
        f2 = 2
        f1 = 1
        for i in range(3, n + 1):
            t = f2 + f1
            f2, f1 = t, f2
        return t
