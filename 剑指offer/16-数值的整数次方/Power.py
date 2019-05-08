"""
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
"""

# -*- coding:utf-8 -*-
class Solution:
    def Power1(self, base, exponent):
        # write code here
        result = 1
        for i in range(abs(exponent)):  # 当 exponent=0 时，range(0)为空，不会执行循环，直接返回 result=1
            result *= base
        if exponent < 0:
            return 1/result
        else:
            return result

    def Power2(self, base, exponent):  # 优化方法
        if not exponent:  # 这里定义 0 的 0 次方为 1
            return 1
        if not base:
            return 0
        result = self.Power2(base, abs(exponent)>>1)
        result *= result
        if abs(exponent) & 1:  # 偶数的二进制最后一个数字必为 1, 与 1 位与结果为 1 的数是奇数
            result *= base
        if exponent < 0:
            result = 1/result
        return result

if __name__ == "__main__":
    f = Solution()
    print(f.Power2(0, 0))

