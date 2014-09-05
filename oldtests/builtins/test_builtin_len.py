import numpy as np
from numba import jit

arr = np.array([object() for i in range(10)])
@jit
def f(arr):
    return arr[0]

assert f(arr) == arr[0]
