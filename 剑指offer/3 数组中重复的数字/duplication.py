# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here

        if not numbers:
            return False
        for num in numbers:  # 防止越界
            if num < 0 or num > len(numbers):
                return False

        for i in range(len(numbers)):
            while numbers[i] != i:
                if numbers[i] == numbers[numbers[i]]:
                    duplication[0] = numbers[i]
                    return True
                numbers[numbers[i]], numbers[i] = numbers[i], numbers[numbers[i]]
        return False


if __name__ == "__main__":
    # 测试用例
    numberss = [[2,3,1,0,2,5,3],
               [-1, -2, -3],
               [],
               [1,2,3,0],
               [7,6,5,5]]

    for numbers in numberss:
        duplication = [False]
        Solution().duplicate(numbers,duplication)
        result = duplication[0]
        print(result)