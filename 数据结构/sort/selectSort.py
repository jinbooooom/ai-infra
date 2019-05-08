# -*- coding:utf-8 -*-
import random

class Sort:
    def selectSort(self, l):  # python 风格
        if not l:
            return l
        m = min(l)
        l.remove(m)
        l.insert(0, m)
        return l[:1] + self.selectSort(l[1:])

if __name__ == "__main__":
    f = Sort()
    l = [random.randint(0, 99) for _ in range(10)]
    print(l)
    print(f.selectSort(l))