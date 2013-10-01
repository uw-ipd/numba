# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest
from numba import jit, float_, double

def f(x):
    return x

def g(x):
    if x > 2.0:
        x = None
    return x

def h(x):
    if x == None or x > 0.0:
        return x
    return None

def h2(x, y):
    first = None if x < 0.0 else x
    second = None if y < 0.0 else y
    return [first, second]

class TestFloatOrNone(unittest.TestCase):

    def test_float(self):
        f_float = jit(float_(float_))(f)
        assert f_float(None) == None
        assert f_float(2.0) == 2.0

    def test_double(self):
        f_double = jit(double(double))(f)
        assert f_double(None) == None
        assert f_double(2.0) == 2.0

    def test_float_assign_none(self):
        assign = jit(double(double))(g)
        assert assign(3.0) == None
        assert assign(1.0) == 1.0

    def test_return_double_or_none(self):
        h_func = jit(double(double))(h)
        assert h_func(1.0) == 1.0
        assert h_func(-1.0) == None
        assert h_func(None) == None

    def test_return_fixed_len_list_double_or_none(self):
        h2_func = jit(double[:](double, double))(h2)
        assert h2_func(2.0, 3.0) == [2.0, 3.0]
        assert h2_func(2.0, -3.0) == [2.0, None]
        assert h2_func(-2.0, -3.0) == [None, None]
        assert h2_func(-2.0, 3.0) == [None, 3.0]

if __name__ == '__main__':
    unittest.main()

