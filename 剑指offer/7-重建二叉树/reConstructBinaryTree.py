"""
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
"""

# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if not pre:
            return None
        if len(pre) == 1:
            return TreeNode(pre[0])
        tree = TreeNode(pre[0])
        p = tin.index(pre[0])  # 下标 p 将 tin 分成左子树和右子树两部分
        # 找到每一个子数的 root 节点的下标，再将其分为左子树和右子树
        tree.left = self.reConstructBinaryTree(pre[1:p+1], tin[:p])
        tree.right = self.reConstructBinaryTree(pre[p+1:], tin[p+1:])
        return tree

