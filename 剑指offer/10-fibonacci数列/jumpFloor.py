"""一只青蛙一次可以跳上1级台阶，也可以跳上2级。
求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。 """

"""
若最后一次跳只跳两个台阶则有f(n-1)种方法
若最后一次跳只跳两个台阶则有f(n-2)中方法"""
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, n):
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

    """  
    def jumpFloor2(self, n):

        if n <= 3 and n >= 0:
            return n
        if n > 3:
            f3 = 3
            f2 = 2
            for i in range(4, n + 1):
                t = f3 + f2
                f3, f2 = t, f3
            return t
    """

