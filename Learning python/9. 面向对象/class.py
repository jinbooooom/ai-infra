# -*- coding:utf-8 -*-

class Person(object):
    def __init__(self, name, age, job=None, pay=0):
        self.__name = name
        self.__age = age
        self.__job = job
        self.__pay = pay

    def salary(self):
        return self.__pay

    def giveRaise(self, percent):
        self.__pay = int(self.__pay * (1 + percent))

    def __repr__(self):
        return "{}'s salary is {}".format(self.__name, self.salary())


class Manager(Person):
    # 可以同时继承多个父类[P850图29-1]，多继承搜索顺序https://www.cnblogs.com/panyinghua/p/3283726.html

    def __init__(self, name, age, pay):  # 定义新的构造函数
        Person.__init__(self, name, age, 'Manager', pay)

    def giveRaise(self, percent, bonus=.1):  # 重载父类中的方法
        # self.__pay = int(self.__pay * (1 + percent + bonus)) # 当改变涨薪水的方式，就需要在两个地方改变代码
        Person.giveRaise(self, percent + bonus)  # 这样只需要改变Person类中的涨薪水代码
        # 类的调用直接有效地避开了继承地限制。从而能将调用直接定位到类树上的某一个特定版本[P816]


if __name__ == "__main__":
   pass