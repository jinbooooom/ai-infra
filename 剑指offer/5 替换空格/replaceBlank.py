# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        t = ''
        for i in s:
            if i == ' ':
                i = '%20'
            t += i
        return t
