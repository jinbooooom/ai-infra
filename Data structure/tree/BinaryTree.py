# -*- coding:utf-8 -*-


class BinaryTree:

    def __init__(self, rootObj):
        self.key = rootObj
        self.left = None
        self.right = None

    def insertLeft(self, newNode):
        if not self.left:  # 若左孩子节点为空
            self.left = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.left = self.left
            self.left = t

    def insertRight(self, newNode):
        if not self.right:
            self.right = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.right = self.right
            self.right = t

    def getLeft(self):  # 指针
        return self.left

    def getRight(self):  # 指针
        return self.right

    def setRootVal(self, obj):
        self.key == obj

    def getRootVal(self):
        return self.key

    def preorder(self, tree):  # 前序遍历
        if tree:
            print(tree.getRootVal())
            self.preorder(tree.getLeft())
            self.preorder(tree.getRight())

    def inorder(self, tree):  # 中序遍历
        if tree:
            self.inorder(tree.getLeft())
            print(tree.getRootVal())
            self.inorder(tree.getRight())

    def posorder(self, tree):  # 后序遍历
        if tree:
            self.posorder(tree.getLeft())
            self.posorder(tree.getRight())
            print(tree.getRootVal())

if __name__ == "__main__":
    tree = BinaryTree('root')
    tree.insertLeft('2a')
    tree.insertRight('2b')
    print(tree.getRootVal())
    print(tree.getLeft().getRootVal())
    print(tree.getRight().getRootVal())
    tree.getLeft().insertLeft('3a')
    tree.getLeft().insertRight('3b')
    tree.getRight().insertLeft('3c')
    print('前序遍历:')
    tree.preorder(tree)
    print('中序遍历:')
    tree.inorder(tree)
    print('后序遍历:')
    tree.posorder(tree)