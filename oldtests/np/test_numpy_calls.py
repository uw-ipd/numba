
import numpy as np

from numba.decorators import jit, autojit

a = np.arange(80).reshape(8, 10)

@autojit
def np_sum(a):
    return np.sum(a, axis=0)

@autojit
def np_copy(a):
    return a.copy(order='F')

@autojit
def attributes(a):
    return (a.T,
            a.T.T,
            a.copy(),
            np.array(a, dtype=np.double))

def test_numpy_attrs():
    result = np_sum(a)
    np_result = np.sum(a, axis=0)
    assert np.all(result == np_result)
    if np.__version__ >= '1.6':
        assert np_copy(a).strides == a.copy(order='F').strides
    assert all(np.all(result1 == result2)
                   for result1, result2 in zip(attributes(a),
                                               attributes.py_func(a)))

if __name__ == "__main__":
    test_numpy_attrs()
