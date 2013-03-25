from ctypes import PYFUNCTYPE, py_object
from numba import translate, codegen, execution, llvm_passes
from timeit import default_timer as timer

def test1(a, b, c):
    return a + b * c / 4

def test2(a, b, c):
    if a > b:
        c += a
    else:
        c += b
    return c

def test3(a, b, c):
    for i in range(1, a):
        c += i
    return c

def test4(a, b, c):
    for i in range(1, a):
        c += i
        for j in range(b):
            c += j
    return c


def main():
    test = test4

    trans = translate.translate(test)

    translate.remove_redundant_refct(trans.lmod)
    codegen.lower(trans.lmod)
    codegen.inline_all_abstract(trans.lmod)
    codegen.error_handling(trans.lmod)

    pm = llvm_passes.make_pm()
    pm.run(trans.lmod)
    print trans.lmod

    exe = execution.ExecutionContext(trans.lmod)
    callable = exe.prepare(trans.lfunc)
    a, b, c = 1000, 2000, 30

    exp = test(a, b, c)

    got = callable(a, b, c)

    print got
    print exp
    assert exp == got


if __name__ == '__main__':
    main()
