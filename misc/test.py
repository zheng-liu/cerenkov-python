def sum(a,b):
    return a+b

def divide(a,b):
    return a*1.0/b


# x = [1,2]
# print sum(*x)
l = [sum, divide]

from itertools import repeat
# print [x for x in repeat(func, 2) for func in l]
print [x for x in l]