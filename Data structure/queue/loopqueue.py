# -*- coding:utf-8 -*-


class loopQueue(object):
    """
    用列表实现循环队列
    """
    def __init__(self, size=10):
        self.items = [None] * (size + 1)
        # 牺牲一个存储空间，front 前面不存数据，当 rear 在 front 前面的时候就是满了
        self.front = 0
        self.rear = 0

        self.realSize = size + 1  # 队列实际的长度，该长度比元素个数大1

    def __len__(self):  # 队列中元素个数
        return self.realSize - 1

    def __str__(self):
        return str(self.getItems())

    def enqueue(self, item):
        if self.isFull():
            print("队列已满，不能入队\n")
            return -1
        self.items[self.rear] = item
        self.rear = (1 + self.rear) % self.realSize  # 保证rear取值范围在[0，self.realSize]

    def dequeue(self):
        if self.isEmpty():
            print("队列为空，不能出队\n")
        result = self.items[self.front]
        self.front = (1 + self.front) % self.realSize  # 保证front取值范围在[0，self.realSize]
        return result

    def isEmpty(self):
        return self.front == self.rear

    def isFull(self):
        if self.front > self.rear:
            return self.front - self.rear == 1
        else:
            return self.front + self.realSize - self.rear == 1

    def getItems(self):
        if self.front < self.rear:
            return self.items[self.front: self.rear]
        elif self.front > self.rear:
            return self.items[self.front:] + self.items[:self.rear]
        else:  # 此时为空
            return []

if __name__ == "__main__":
    q = loopQueue(5)  # 初始化循环队列元素个数为 5
    print(q)
    print(q.isEmpty())
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    print(q)
    print(q.isFull())
    q.enqueue(4)
    q.enqueue(5)
    print(q)
    print(q.isFull())
    q.enqueue(6)
    print(q)
    print(q.dequeue())
    print(q.dequeue())
    print(q)
    q.enqueue(7)
    q.enqueue(8)
    print(q)
    print(len(q))
    print(q.isFull())
