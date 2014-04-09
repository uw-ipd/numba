import numpy as np
import operator
from pylab import imshow, show
from alge import Case, of


class Operation(object):
    pass

class UnaryOperation(Operation):

    def __init__(self, operand, op):
        self.operand = operand
        self.op = op

    @property
    def value(self):
        return self.op(self.operand.value)
    
class BinaryOperation(Operation):

    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op

    @property
    def value(self):
        return self.op(self.left.value, self.right.value)
    
class AssignOperation(Operation):

    def __init__(self, original, key, rhs):
        self.original = original
        self.key = key
        self.rhs = rhs

    @property
    def value(self):
        operator.setitem(self.original.value, self.key.value, self.rhs.value)
        return self.original.value


def unary_op(op):
    
    def wrapper(func):
        def impl(self):
            return Array(operation=UnaryOperation(self, op))

        return impl

    return wrapper


def binary_op(op):
    
    def wrapper(func):
        def impl(self, other):
            if not isinstance(other, Array) and not isinstance(other, Operation):
                other = Array(other)
            return Array(operation=BinaryOperation(self, other, op))

        return impl

    return wrapper


class Array(object):
    
    def __init__(self, data=None, operation=None):
        
        self.data = data
        self.operation = operation

    @property
    def value(self):
        if self.data is None:
            if self.operation is None:
                raise ValueError("Don't know how to compute value")
            self.data = self.operation.value
        return self.data

    @binary_op(operator.add)
    def __add__(self, other):
        pass

    @binary_op(operator.sub)
    def __sub__(self, other):
        pass

    @binary_op(operator.mul)
    def __mul__(self, other):
        pass

    @binary_op(operator.div)
    def __div__(self, other):
        pass

    @binary_op(operator.le)
    def __le__(self, other):
        pass

    @binary_op(operator.pow)
    def __pow__(self, other):
        pass

    @binary_op(operator.getitem)
    def __getitem__(self, other):
        pass

    def __setitem__(self, key, value):
        if not isinstance(key, Array) and not isinstance(key, Operation):
            key = Array(data=key)
        if not isinstance(value, Array) and not isinstance(value, Operation):
            value = Array(data=value)

    def copy(self):
        pass
        

@unary_op(np.abs)
def numba_abs(operand):
    pass


if __name__ == '__main__':

    a1 = Array(data=np.arange(-10,10))
    a2 = Array(data=np.arange(-10,10))
    result = a1**2 + numba_abs(a2)
    print result.value

