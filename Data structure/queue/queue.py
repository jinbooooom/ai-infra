# -*- coding:utf-8 -*-

class Queue:
    """先进先出"""
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def isEmpty(self):
        return self.items == []

    def size(self):
        return len(self.items)

    def clear(self):  # 设置为空队列
        del self.items[:]

if __name__ == "__main__":
    q = Queue()
    q.enqueue(9)
    q.enqueue(8)
    q.enqueue(5)
    print(q.items)
    print(q.size())
    print(q.isEmpty())
    q.dequeue()
    print(q.items)
    print(q.size())
    print(q.isEmpty())
    q.clear()
    print(q.items)
    print(q.size())
    print(q.isEmpty())
