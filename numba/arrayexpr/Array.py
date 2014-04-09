import numpy as np
import operator
from pylab import imshow, show
from alge import Case, of, datatype
from collections import namedtuple


UnaryOperation = datatype('UnaryOperation', ['operand', 'op', 'op_str'])
BinaryOperation = datatype('BinaryOperation', ['lhs', 'rhs', 'op', 'op_str'])


def unary_op(op, op_str):
    
    def wrapper(func):
        def impl(self):
            return Array(operation=UnaryOperation(self, op, op_str))

        return impl

    return wrapper


def binary_op(op, op_str):
    
    def wrapper(func):
        def impl(self, other):
            if not isinstance(other, Array):
                other = Array(other)
            return Array(operation=BinaryOperation(self, other, op, op_str))

        return impl

    return wrapper


class Array(namedtuple('Array', ['data', 'operation'])):

    def __new__(cls, data=None, operation=None):
        return super(Array, cls).__new__(cls, data, operation)

    def __str__(self):
        return str(Value(self))
    
    @binary_op(operator.add, 'operator.add')
    def __add__(self, other):
        pass

    @binary_op(operator.sub, 'operator.sub')
    def __sub__(self, other):
        pass

    @binary_op(operator.mul, 'operator.mul')
    def __mul__(self, other):
        pass

    @binary_op(operator.div, 'operator.div')
    def __div__(self, other):
        pass

    @binary_op(operator.le, 'operator.le')
    def __le__(self, other):
        pass

    @binary_op(operator.pow, 'operator.pow')
    def __pow__(self, other):
        pass

    @binary_op(operator.getitem, 'operator.getitem')
    def __getitem__(self, other):
        pass

    def __setitem__(self, key, value):
        if not isinstance(key, Array):
            key = Array(data=key)
        if not isinstance(value, Array):
            value = Array(data=value)

    def copy(self):
        pass
        

@unary_op(np.abs, 'abs')
def numba_abs(operand):
    pass


class Value(Case):

    @of('Array(data, operation)')
    def array(self, data, operation):
        if data is not None:
            return data
        else:
            return Value(operation)

    @of('UnaryOperation(operand, op, op_str)')
    def unary_operation(self, operand, op, op_str):
        return op(Value(operand))

    @of('BinaryOperation(lhs, rhs, op, op_str)')
    def binary_operation(self, lhs, rhs, op, op_str):
        return op(Value(lhs), Value(rhs))


class CodeGen(Case):

    @of('Array(data, operation)')
    def array(self, data, operation):
        if data is not None:
            return str(data)
        else:
            return str(CodeGen(operation))

    @of('UnaryOperation(operand, op, op_str)')
    def unary_operation(self, operand, op, op_str):
        print op, CodeGen(operand)

    @of('BinaryOperation(lhs, rhs, op, op_str)')
    def binary_operation(self, lhs, rhs, op, op_str):
        print op, CodeGen(lhs), CodeGen(rhs)


if __name__ == '__main__':

    a1 = Array(data=np.arange(-10,10))
    a2 = Array(data=np.arange(-10,10))
    result = a1**2 + numba_abs(a2)
    #CodeGen(result)
    print result

