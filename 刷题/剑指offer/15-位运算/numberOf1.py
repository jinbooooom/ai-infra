# -*- coding:utf-8 -*-

"""
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
有必要了解对负数，右移左边补1
对正数右移，左边补零
"""

class Solution:
    def NumberOf1(self, n):
        # write code here
        return sum([(n>>i & 1) for i in range(32)])
        """
        对负数会引起死循环
        count = 0
        while n:
            count += (n & 1)
            n = n>>1
        return count
        """

if __name__ == "__main__":
    f = Solution()
    print(f.NumberOf1(-11))