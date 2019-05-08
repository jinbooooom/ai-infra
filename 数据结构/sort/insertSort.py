# -*- coding:utf-8 -*-
import random

class Sort:
    def insertSort(self, l):
        for i in range(len(l) - 1):  # 插入 len(l)-1 次就排序完毕
            t = l[i + 1]
            for j in range(i, -1, -1):  # i～0
                if l[j] >= t:
                    l[j + 1] = l[j]
                else:
                    l[j + 1] = t
                    break
                l[j] = t
        return l

    def insertSort2(self, l):  # python 风格
        for i in range(len(l) - 1):
            t = l[i + 1]
            for j in range(i + 1):
                if l[j] >= t:
                    del l[i + 1]
                    l.insert(j, t)
                    break
        return l


if __name__ == "__main__":
    l = [9, 8, 5, 6, 3]
    f = Sort()
    l = [random.randint(0, 99) for _ in range(10)]
    print(l)
    print(f.insertSort(l))
    print(bool(set(f.insertSort(l)) == set(l)))
    print(f.insertSort2(l))
    print(bool(set(f.insertSort2(l)) == set(l)))
    for i in range(5, -1, -1):
        print(i)