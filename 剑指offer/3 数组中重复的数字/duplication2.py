# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here

        if not numbers:
            return False
        for num in numbers:  # 防止越界
            if num < 1 or num > len(numbers):
                return False

        start = 1
        end = len(numbers) - 1
        while end >= start:
            mid = ((end - start) >> 1) + start
            count = self.countRange(numbers, start, mid)
            if start == end:
                if count > 1:
                    duplication[0] = start
                    return True
            if count > mid - start + 1:
                end = mid
            else:
                start = mid + 1
        return False

    def countRange(self, numbers, start, end):
        if not numbers:
            return False
        count = 0
        for number in numbers:
            if number >= start and number <= end:
                count += 1
        return count

if __name__ == "__main__":
    # 测试用例
    numberss = [[2, 3, 5, 4, 3, 2, 6, 7],
               [-1, -2, -3],
               [],
               [1, 2, 4, 4, 3],
               [7, 6, 5, 5]]

    for numbers in numberss:
        duplication = [False]
        Solution().duplicate(numbers, duplication)
        print(duplication[0])