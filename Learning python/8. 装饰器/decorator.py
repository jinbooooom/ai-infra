## -*- coding:utf-8 -*-


# import functools            #这句不懂可以看廖老师的教程
import time


def metric(fn):
    #@functools.wraps(fn)        #这句不懂可以看廖老师的教程
    def wrapper(*args, **kw):
        start = time.time()
        res = fn(*args, **kw)
        print('%s executed in %f s' % (fn.__name__, time.time()-start))
        return res
    return wrapper
#以下是测试效果
@metric
def test1():
    time.sleep(2)
test1()

@metric
def test2():
    time.sleep(3)
test2()
