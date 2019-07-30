# -*- coding:utf-8 -*-

class Stack:
    """先进后出"""
    def __init__(self):
        self.items = []

    def push(self, item):  # 压入
        self.items.append(item)

    def pop(self):  # 弹出
        return self.items.pop()

    def clear(self):
        del self.items[:]

    def isEmpty(self):  # 判断是否为空
        return self.items == []

    def size(self):
        return len(self.items)

    def peek(self):  # 返回 stack 顶部元素，但不会修改 stack
        return self.items[self.size()-1]


if __name__ == "__main__":
    s = Stack()
    s.push(8)
    s.push(5)
    s.push(9)
    print(s.items)
    print(s.size())
    print(s.pop())
    print(s.items)
    print(s.isEmpty())
    print(s.peek())
    s.clear()
    print(s.items)
    print(s.isEmpty())


