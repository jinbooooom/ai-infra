class Solution:
    def Fibonacci1(self, n):
        # write code here
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        if n > 2:
            return self.Fibonacci1(n-1) + self.Fibonacci1(n-2)

    def Fibonacci2(self, n):
        # write code here
        if n == 0:
            return 0
        if n == 1 or n == 2:
            return 1
        if n > 2:
            a = b = 1
            for i in range(3, n + 1):
                t = a + b
                a, b = t, a
            return t


s = Solution()
l = []
for i in range(9):
    l.append(s.Fibonacci2(i))
print(l)