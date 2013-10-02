# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import unittest
from numba import jit, float_, double

def identity(x):
    return x

def assign(x):
    if x > 2.0:
        x = None
    return x

def return_scalar(x):
    if x == None or x > 0.0:
        return x
    return None

def return_list(x, y):
    first = None if x < 0.0 else x
    second = None if y < 0.0 else y
    return [first, second]

class TestFloatOrNone(unittest.TestCase):

    def test_float(self):
        identity_float = jit(float_(float_))(identity)
        assert identity_float(None) == None
        assert identity_float(2.0) == 2.0

    def test_double(self):
        identity_double = jit(double(double))(identity)
        assert identity_double(None) == None
        assert identity_double(2.0) == 2.0

    def test_assign_float_none(self):
        assign_func = jit(float_(float_))(assign)
        assert assign_func(3.0) == None
        assert assign_func(1.0) == 1.0

    def test_assign_double_none(self):
        assign_func = jit(double(double))(assign)
        assert assign_func(3.0) == None
        assert assign_func(1.0) == 1.0

    def test_return_scalar_float(self):
        return_scalar_func = jit(float_(float_))(return_scalar)
        assert return_scalar_func(1.0) == 1.0
        assert return_scalar_func(-1.0) == None
        assert return_scalar_func(None) == None

    def test_return_scalar_double(self):
        return_scalar_func = jit(double(double))(return_scalar)
        assert return_scalar_func(1.0) == 1.0
        assert return_scalar_func(-1.0) == None
        assert return_scalar_func(None) == None

    def test_return_list_float(self):
        return_list_func = jit(float_[:](float_, float_))(return_list)
        assert return_list_func(2.0, 3.0) == [2.0, 3.0]
        assert return_list_func(2.0, -3.0) == [2.0, None]
        assert return_list_func(-2.0, -3.0) == [None, None]
        assert return_list_func(-2.0, 3.0) == [None, 3.0]

    def test_return_list_double(self):
        return_list_func = jit(double[:](double, double))(return_list)
        assert return_list_func(2.0, 3.0) == [2.0, 3.0]
        assert return_list_func(2.0, -3.0) == [2.0, None]
        assert return_list_func(-2.0, -3.0) == [None, None]
        assert return_list_func(-2.0, 3.0) == [None, 3.0]

if __name__ == '__main__':
    unittest.main()

