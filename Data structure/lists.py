# -*- coding:utf-8 -*-
class Node(object):
    def __init__(self, value):
        # 元素域
        self.value = value
        # 链接域
        self.next = None

class UnorderedList(object):  # 单向线性表
    """reference:https://www.cnblogs.com/yifeixu/p/8954991.html"""
    def __init__(self, node=None):
        self.__head = node

    def __len__(self):  # size
        # 游标 cursor ，用来遍历链表
        cur = self.__head
        # 记录遍历次数
        count = 0
        # 当前节点为None则说明已经遍历完毕
        while cur:
            count += 1
            cur = cur.next
        return count

    def isEmpty(self):
        # 头节点不为None则不为空
        return self.__head == None

    def add(self, value):  # 链表头部添加元素
        """
        头插法
        先让新节点的next指向头节点
        再让头节点指向新节点
        顺序不可错，要先保证原链表的链不断，否则头节点后面的链会丢失
        """
        node = Node(value)
        node.next = self.__head
        self.__head = node  # node是最新插入节点的地址

    def append(self, value):  # 链表尾部添加元素
        """尾插法"""
        node = Node(value)
        cur = self.__head
        if self.isEmpty():
            self.__head = node
        else:
            while cur.next:
                cur = cur.next
            cur.next = node

    def insert(self, pos, value):  # 指定位置添加元素
        # 应对特殊情况:插入位置小于 0 就使用头插法，插入位置溢出长度就采用尾插法
        if pos <= 0:
            self.add(value)
        elif pos > len(self) - 1:
            self.append(value)
        else:
            node = Node(value)
            prior = self.__head
            count = 0
            # 在插入位置的前一个节点停下
            while count < (pos - 1):
                prior = prior.next
                count += 1
            # 先将插入节点与节点后的节点连接，防止链表断掉，先链接后面的，再链接前面的
            node.next = prior.next
            prior.next = node

    def remove(self, value):  # 删除节点
        cur = self.__head
        prior = None
        while cur:
            if value == cur.value:
                # 判断此节点是否是头节点
                if cur == self.__head:
                    self.__head = cur.next
                else:
                    prior.next = cur.next
                break
            # 还没找到节点，有继续遍历
            else:
                prior = cur
                cur = cur.next

    def search(self, value):  # 查找节点是否存在
        cur = self.__head
        while cur:
            if value == cur.value:
                return True
            cur = cur.next
        return False

    def traverse(self):  # 遍历整个链表
        cur = self.__head
        while cur:
            print(cur.value)
            cur = cur.next

if  __name__ == "__main__":
    l = UnorderedList()
    print(l.isEmpty())