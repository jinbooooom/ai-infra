# -*- coding:utf-8 -*-
"""
对一个二维数组，删除其中的若干行，比如第i,j,k行;删除其中的若干列，比如第m,n,l列.
返回剩下的数组。输入要删除的行为[i,j,k]，要删除的列为[m,n,l]

"""
import numpy as np

def delRow(drow, array):
    drow = sorted(drow)
    t = drow[0] + 1
    while t in drow:
        t += 1
    if t >= len(array):
        return array[:len(array) - len(drow)]
    array[drow[0]] = array[t]
    drow = drow[1:]+[t]
    #print('drow', drow, 'array:', array)
    return delRow(drow, array)

def delArray(drow, dcol, array):
    if drow:
        array = delRow(drow, array)
        print(array)
    #print("-" * 50)
    if dcol:
        array = delRow(dcol, array.T)
        print(array.T)
    return array.T

"""
def delCol(dcol, array):
    t = dcol[0] + 1
    while t in dcol:
        t += 1
    if t >= array.shape[1]: # 如果这t >= 列数，返回
        #return array[:, :(array.shape[1] - len(dcol))]
        pass
    array[:, :dcol[0]] = array[:, :t]
    dcol = sorted([dcol[1], t])
    print('drow', drow, 'array:', array)
    return delCol(drow, array)
"""

if __name__ == "__main__":
    a = np.array(
        [[1, 2, 3, 4],
         [4, 5, 6, 7],
         [7, 8, 9, 10],
         [5, 9, 5, 5],
         [6, 9, 6, 6],
         [8, 8, 8, 8]])

    drow = [4,2]
    dcol = [3,2]
    #print(delRow(drow, a))
    delArray(drow, dcol, a)
