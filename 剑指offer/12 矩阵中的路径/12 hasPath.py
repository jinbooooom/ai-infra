"""
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。
例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，
但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
"""

# -*- coding:utf-8 -*-
class Solution:
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        # 输入并不是二维列表
        if not matrix or not path or rows < 1 or cols < 1:
            return False
        visited = [0] * (rows * cols)
        pathLength = 0
        for i in range(rows):
            for j in range(cols):
                if self.hasPathCore(matrix, rows, cols, i, j, path, pathLength, visited):
                    return True
        return False

    def hasPathCore(self, matrix, rows, cols, i, j, path, pathLength, visited):
        if len(path) == pathLength:
            return True
        hasPath = False
        if i >= 0 and i < rows and j >= 0 and j < cols and matrix[i * cols + j] == path[pathLength] and not visited[
            i * cols + j]:
            pathLength += 1
            visited[i * cols + j] = 1
            hasPath = self.hasPathCore(matrix, rows, cols, i, j - 1, path, pathLength, visited) or \
                      self.hasPathCore(matrix, rows, cols, i - 1, j, path, pathLength, visited) or \
                      self.hasPathCore(matrix, rows, cols, i, j + 1, path, pathLength, visited) or \
                      self.hasPathCore(matrix, rows, cols, i + 1, j, path, pathLength, visited)
            if not hasPath:
                pathLength -= 1
                visited[i * cols + j] = 0
        return hasPath