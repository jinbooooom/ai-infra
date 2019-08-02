# -*- coding:utf-8 -*-


class Deque:
    def __init__(self):
        self.items = []

    def addFront(self, item):  # 首部添加元素
        self.items.append(item)

    def addRear(self, item):  # 尾部添加元素
        self.items.insert(0, item)

    def removeFront(self):  # 首部删除元素
        return self.items.pop()

    def removeRear(self):  # 尾部删除元素
        return self.items.pop(0)

    def size(self):
        return len(self.items)

    def isEmpty(self):
        return self.items == []

    def clear(self):
        del self.items[:]

if __name__ == "__main__":
    d = Deque()
    d.addRear(5)
    d.addRear(6)
    d.addRear(7)
    d.addFront(8)
    d.addFront(9)
    d.addFront(10)
    print(d.items)
    d.removeFront()
    print(d.items)
    d.removeRear()
    print(d.items)
    d.clear()
    print(d.items)
    print(d.isEmpty())


