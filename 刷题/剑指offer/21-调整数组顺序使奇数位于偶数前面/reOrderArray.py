"""
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
并保证奇数和奇数，偶数和偶数之间的相对位置不变。
"""


# -*- coding:utf-8 -*-
class Solution:
    # 类似冒泡排序，前面一个数为偶后面一个数为奇就调换位置
    def reOrderArray(self, array):
        # write code here
        if not array:
            return array
        for i in range(len(array) - 1):
            for j in range(len(array) - i - 1):
                if not self.isOdd(array[j]) and self.isOdd(array[j + 1]):
                    array[j], array[j + 1] = array[j + 1], array[j]
        return array

    def reOrderArray2(self, array):
        if not array:
            return array
        l_odd = []
        l_even = []
        for i in range(len(array)):
            if self.isOdd(array[i]):
                l_odd.append(array[i])
            else:
                l_even.append(array[i])
        return l_odd + l_even

    def reOrderArray3(self, array):  # 没有考虑相对位置不变
        # write code here
        if not array:
            return array
        p1 = 0
        p2 = len(array) - 1
        while p2 > p1:
            # 只要 p1 指向奇数就自增，即指向偶数就停下来
            while self.isOdd(array[p1]) and p1 < p2:
                p1 += 1
            # 只要 p2 指向偶数就自增，即指向奇数就停下来
            while not self.isOdd(array[p2]) and p1 < p2:
                p2 -= 1
            array[p1], array[p2] = array[p2], array[p1]
        return array

    def isOdd(self, d):
        return d & 1