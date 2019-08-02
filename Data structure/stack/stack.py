# -*- coding:utf-8 -*-


class Stack:
    """先进后出"""
    def __init__(self):
        self.items = []

    def push(self, item):  # 压入
        self.items.append(item)

    def pop(self):  # 弹出
        return self.items.pop()

    def isEmpty(self):  # 判断栈是否为空
        return self.items == []

    def size(self):
        return len(self.items)

    def peek(self):  # 返回 stack 顶部元素，但不会修改 stack
        return self.items[-1]

    def clear(self):  # 设置为空栈
        del self.items[:]



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
    print(s.items)
    s.clear()
    print(s.items)
    print(s.isEmpty())


