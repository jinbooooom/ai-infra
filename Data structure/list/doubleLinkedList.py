# -*- coding:utf-8 -*-
class Node(object):
    def __init__(self, item):
        # 元素域
        self.item = item
        # 链接域
        self.prior = None
        self.next = None


class DoubleLinkedList(object):  # 双向链表
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

    """
    用时再补充
    """

if  __name__ == "__main__":
    pass
