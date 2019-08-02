# -*- coding:utf-8 -*-

"""
参考博客：https://www.jianshu.com/p/bae9e0b0a91c
"""
class LoopQueue(object):
    def __init__(self, n=10):
        self.arr = [None] * (n+1)  # 由于特意浪费了一个空间，所以arr的实际大小应该是用户传入的容量+1
        self.front = 0
        self.tail = 0
        self.size = 0

    def __str__(self):
        return str(self.arr)

    def __len__(self):
        return len(self.arr)

    def __iter__(self):
        return iter(self.arr)

    def get_size(self):
        # 获取队列元素个数
        return self.size

    def get_capaticty(self):
        # 获取队列容积（实际可存储元素个数）
        return self.__len__() - 1

    def is_full(self):
        # 判断队列是否为满
        return (self.tail+1) % len(self.arr) == self.front

    def is_empty(self):
        # 判断队列是否为空
        return self.size == 0

    def get_front(self):
        # 获取队首
        return self.arr[self.front]

    def enqueue(self, e):
        # 入队
        if self.is_full():
            self.resize(self.get_capaticty() * 2)  # 如果队列满，以当前队列容积的2倍进行扩容
        self.arr[self.tail] = e
        self.tail = (self.tail+1) % len(self.arr)
        self.size += 1

    def dequeue(self):
        # 出队
        if self.is_empty():
            raise Exception("Cannot dequeue from en empty queue")

        result = self.arr[self.front]
        self.arr[self.front] = None
        self.front = (self.front+1) % len(self.arr)
        self.size -= 1

        # 如果元素个数少于容积的1/4并且元素个数大于1
        if self.size < self.get_capaticty() // 4 and self.get_capaticty() > 1:
            self.resize(self.get_capaticty() // 2)
        return result

    def resize(self, new_capacity):
        new_arr = [None] * (new_capacity+1)
        for i in range(self.size):
            new_arr[i] = self.arr[(i+self.front) % len(self.arr)]

        self.arr = new_arr
        self.front = 0
        self.tail = self.size

