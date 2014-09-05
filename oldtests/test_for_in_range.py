"""
>>> test_modify()
0
1
2
3
4
<BLANKLINE>
(4, 0)

>>> test_negindex()
6
5
4
3
2
(2, 0)

>>> test_negindex_inferred()
5
4
3
2
(2, 0)

>>> test_fix()
0
1
2
3
4
<BLANKLINE>
4

>>> test_break()
0
1
2
<BLANKLINE>
(2, 0)

>>> test_else_clause1()
0
1
2

>>> test_else_clause2()
0
1
2
else clause

>>> test_else_clause3()
0
1
2
else clause

>>> test_else_clause4()
inner 0
i 0
else clause 1 0 9
i 1
else clause 2 0 9
i 2
else clause 3 0 9
i 3
else clause 4 0 9
i 4
else clause 5 0 9
i 5
else clause 6 0 9
i 6
else clause 7 0 9
i 7
else clause 8 0 9
i 8
else clause 9 0 9
i 9
else clause

>>> test_return()
0
1
2
(2, 0)


>>> test_return2()
0
1
2
2
"""

from numba import autojit
import unittest


@autojit
def test_modify():
    n = 5
    for i in range(n):
        print(i)
        n = 0
    print('')
    return i,n

@autojit
def test_negindex():
    n = 5
    for i in range(n+1, 1, -1):
        print(i)
        n = 0
    return i,n

@autojit
def test_negindex_inferred():
    n = 5
    for i in range(n, 1, -1):
        print(i)
        n = 0
    return i,n

@autojit
def test_fix():
    for i in range(5):
        print(i)
    print('')
    return i

@autojit
def test_break():
    n = 5
    for i in range(n):
        print(i)
        n = 0
        if i == 2:
            break
    else:
        print("FAILED!")
    print('')
    return i, n

@autojit
def test_else_clause1():
    for i in range(10):
        if i > 2:
            break
        print(i)
    else:
        print("else clause")

@autojit
def test_else_clause2():
    for i in range(10):
        if i > 2:
            continue
        print(i)
    else:
        print("else clause")

@autojit
def test_else_clause3():
    for i in range(3):
        if i > 2 and i < 2:
            continue
        print(i)
    else:
        print("else clause")

@autojit
def test_else_clause4():
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i == j and j == k:
                    print("inner " + str(i))
                    break
                else:
                    continue
            else:
                print("else clause " + str(i) + ' ' + str(j) + ' ' + str(k))
            break
        else:
            print("else clause " + str(i) + ' ' + str(j))

        print("i " + str(i))
    else:
        print("else clause")

@autojit
def test_return():
    n = 5
    for i in range(n):
        print(i)
        n = 0
        if i == 2:
            return i,n
    print('')
    return "FAILED!"

@autojit
def test_return2():
    n = 5
    for i in range(n):
        print(i)
        n = 0
        for j in range(n):
            return 0
        else:
            if i < 2:
                continue
            elif i == 2:
                for j in range(i):
                    return i
                print("FAILED!")
            print("FAILED!")
        print("FAILED!")
    return -1


if __name__ == '__main__':
    import doctest
    doctest.testmod()
