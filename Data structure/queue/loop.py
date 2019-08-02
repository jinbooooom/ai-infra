# -*- coding:utf-8 -*-


class loopQueue(object):
    def __init__(self, size=10):
        self.arr = [None] * (size + 1)  # 由于特意浪费了一个空间，所以arr的实际大小应该是用户传入的容量+1
        self.front = 0
        self.rear = 0

