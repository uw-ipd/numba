import numpy as np
import operator
from pylab import imshow, show
from alge import Case, of, datatype
from collections import namedtuple


UnaryOperation = datatype('UnaryOperation', ['operand', 'op', 'op_str'])
BinaryOperation = datatype('BinaryOperation', ['lhs', 'rhs', 'op', 'op_str'])
ScalarConstant = datatype('ScalarConstant', ['value'])


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
                other = ScalarConstant(other)
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

    @of('ScalarConstant(value)')
    def scalar_constant(self, value):
        return value

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
            self.state['count'] += 1
            return 'x' + str(self.state['count'])
        else:
            return CodeGen(operation, state=self.state)

    @of('ScalarConstant(value)')
    def scalar_constant(self, value):
        return str(value)

    @of('UnaryOperation(operand, op, op_str)')
    def unary_operation(self, operand, op, op_str):
        return op_str + '(' +  CodeGen(operand, state=self.state) + ')'

    @of('BinaryOperation(lhs, rhs, op, op_str)')
    def binary_operation(self, lhs, rhs, op, op_str):
        return op_str + '(' + CodeGen(lhs, state=self.state) + ',' + CodeGen(rhs, state=self.state) + ')'


def test_mandelbrot():

    width = 900
    height = 600
    x_min = -2.0
    x_max = 1.0
    y_min = -1.0
    y_max = 1.0
    num_iterations = 20

    x, y = np.meshgrid(np.linspace(x_min, x_max, width),
                       np.linspace(y_min, y_max, height))

    c = Array(data = x + 1j*y)
    #z = c.copy()
    z = Array(data = x + 1j*y)

    image = Array(data = np.zeros((height, width)))

    for i in range(num_iterations):

        indices = (np.abs(z) <= 10)
        z[indices] = (z[indices] ** 2) + c[indices]
        image[indices] = i

    imgplot = imshow(np.log(image))
    show()


def test1():

    a1 = Array(data=np.arange(-10,10))
    a2 = Array(data=np.arange(-10,10))

    result = a1**2 + numba_abs(a2)

    print result
    print CodeGen(result, state={'count': 0})


if __name__ == '__main__':
    test1()

