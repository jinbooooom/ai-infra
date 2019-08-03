# -*- coding:utf-8 -*-
class Node(object):
    def __init__(self, item):
        # 元素域
        self.item = item
        # 链接域
        self.next = None


class LinkedList(object):  # 单链表
    def __init__(self, node=None):
        self.head = node

    def __len__(self):  # size
        # 游标 cursor ，用来遍历链表
        cur = self.head
        # 记录遍历次数
        count = 0
        # 当前节点为None则说明已经遍历完毕
        while cur:
            count += 1
            cur = cur.next
        return count

    def add(self, item):  # 链表头部添加元素
        """
        头插法
        先让新节点的next指向头节点
        再让头节点指向新节点
        顺序不可错，要先保证原链表的链不断，否则头节点后面的链会丢失
        """
        node = Node(item)
        node.next = self.head
        self.head = node  # node是最新插入节点的地址

    def append(self, item):  # 链表尾部添加元素
        """尾插法"""
        node = Node(item)
        cur = self.head
        if self.isEmpty():
            self.head = node
        else:
            while cur.next:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):  # 指定位置添加元素,下标从零开始
        # 特殊情况，若 pos <= 0，使用头插法；若 pos >= len(list)，使用尾插法。
        if pos <= 0:
            self.add(item)
        elif pos >= len(self):
            self.append(item)
        else:
            cur = self.head
            node = Node(item)
            while pos - 1:
                # 从 0(self.__head) 到 pos 位置，cur 要移动 pos 次
                # 但只移动 pos - 1 就好，因为前面一个节点能找到后面的节点，后面的节点找不到前面的节点。
                cur = cur.next
                pos -= 1
            # cur 最终指向要插入的位置前面一个节点
            node.next = cur.next
            cur.next = node

    def remove(self, item):  # 删除值等于 item 的节点
        cur = self.head
        prior = None
        while cur:
            if cur.item == item:
                if cur == self.head:  # 如果删除第一个节点
                    self.head = self.head.next
                elif not cur.next:  # 如果删除最后一个节点
                    prior.next = None
                else:
                    prior.next = cur.next
                break  # 此时的 cur 始终为 True，得跳出。
            else:
                prior = cur
                cur = cur.next

    def search(self, item):  # 查找节点是否存在
        cur = self.head
        while cur:
            if cur.item == item:
                return True
            else:
                cur = cur.next
        return False

    def isEmpty(self):
        # 头节点不为 None 则不为空
        return not self.head

    def traverse(self):  # 遍历整个链表
        cur = self.head
        t = []
        while cur:
            t.append(cur.item)
            cur = cur.next
        print(t)

if  __name__ == "__main__":
    L = LinkedList()
    print(L.isEmpty())
    L.append(6)
    L.append(7)
    L.add(4)
    L.add(3)
    L.traverse()
    L.insert(9, 'hello')
    L.insert(2, 5)
    L.traverse()
    print(L.search(5))
    print(L.search(99))
    L.traverse()
    L.remove(4)
    L.traverse()
    L.remove(3)
    L.traverse()
    L.remove('hello')
    L.traverse()
    L.remove(99)
    L.traverse()
    print(len(L))