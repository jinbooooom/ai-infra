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

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def setRootVal(self, obj):
        self.key == obj

    def getRootVal(self):
        return self.key

    def preorder(self, tree):
        if tree:
            print(tree.getRootVal())
            self.preorder(tree.getLeft())
            self.preorder(tree.getRight())

if __name__ == "__main__":
    tree = BinaryTree('root')
    tree.insertLeft('2a')
    tree.insertRight('2b')
    print(tree.getRootVal())
    print(tree.getLeft().getRootVal())
    print(tree.getRight().getRootVal())
    tree.insertLeft('3a')
    print(tree.getLeft().getRootVal())
    print(tree.getLeft().getLeft().getRootVal())