import numba
import numpy as np


@numba.jit
def cast_object(dst_type, value):
    return dst_type(value)


assert cast_object(numba.double, 1) == 1.0

cast_object(numba.double[:], np.arange(10, dtype=np.double))
