# -*- coding:utf-8 -*-

"""
请实现一个函数用来匹配包括'.'和'*'的正则表达式。
模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
在本题中，匹配是指字符串的所有字符匹配整个模式。
例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
"""

class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        # 字符串和模板长度均为零时匹配
        if len(s) == 0 and len(pattern) == 0:
            return True
        # 模板为空而字符串不为空肯定不匹配，但反过来字符串为空模板不为空可能匹配。
        if len(s) > 0 and len(pattern) == 0:
            return False
        if len(pattern) > 1 and pattern[1] == '*':  # 这里考虑了模板包含 '.*' 的情况
            if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.'):
                # 若s[0]已经与模板匹配，则包含有三种情况继续往下比较是否匹配
                # 一、形如 "aab" 与 "a*b" 用 self.match(s[1:], pattern) 继续判断后面的是否匹配
                # 二、形如 "aab" 与 "a*a*b" 用 self.match(s[1:], pattern[2:]) 继续判断后面的是否匹配
                # 三、形如 "aab" 与 "a*a*a*d*d*d*d*a*b" 必须用 self.match(s, pattern[2:]) 跳过一些没有用的模板
                return self.match(s[1:], pattern) or self.match(s[1:], pattern[2:]) or self.match(s, pattern[2:])
            else:
                # 不满足条件说明模板中对应的那个字符在字符串中出现零个，所以后移两位作为新模板
                return self.match(s, pattern[2:])
        else:
            if len(s) > 0 and (s[0] == pattern[0] or pattern[0] == '.'):
                return self.match(s[1:], pattern[1:])
            else:
                return False