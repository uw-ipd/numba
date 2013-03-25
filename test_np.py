import numpy as np
from ctypes import PYFUNCTYPE, py_object
from numba import translate, codegen, execution, llvm_passes
from timeit import default_timer as timer

def test1(a, b, c):
    return a + b * c / 4

def test2(a, b, c):
    for i in range(a.shape[0]):
        a[i] += b[i] + c[i]
    return a

def test3(a, b, c):
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            for k in range(c.shape[0]):
                a[i] += b[j] + c[k]
    return a

def main():
    test = test3

    trans = translate.translate(test)

    translate.remove_redundant_refct(trans.lmod)
    codegen.lower(trans.lmod)
    codegen.inline_all_abstract(trans.lmod)
    codegen.error_handling(trans.lmod)

    pm = llvm_passes.make_pm()
    pm.run(trans.lmod)
    print trans.lmod
#    trans.lfunc.viewCFG()

    exe = execution.ExecutionContext(trans.lmod)
    callable = exe.prepare(trans.lfunc)

    a = np.arange(10)
    b = np.arange(10, 20)
    c = np.arange(20, 30)


    exp = test(a, b, c)

    got = callable(a, b, c)

    print got
    print exp
    assert np.all(exp == got)


if __name__ == '__main__':
    main()
