from numba import translate

def test1(a, b):
    return a + b

def test2(x, y):
    if x > y:
        return x
    else:
        return y

def test3(a, b):
    s = 0
    for i in range(a, b):
        s += i
    return s

def test4(a, b):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            a[i, j] = b[i, j]

def test5(a, b):
    y = 0
    if a > b:
        y = 123
    else:
        y = 1.23
    return y

def test6(a, b):
    s = 0
    while a < b:
        s += 1
    return s

def test7(a, b):
    s = 0
    with nopython:
        while a < b:
            s += 1
        return s

def test8(a):
    ra = range(a)
    j = 0
    for i in ra:
        j += i
    return j


def main():
#    print translate.translate(test1).lmod
#    print translate.translate(test2).lmod
#    print translate.translate(test3).lmod
#    print translate.translate(test4).lmod
#    print translate.translate(test5).lmod
#    print translate.translate(test6).lmod
#    print translate.translate(test7).lmod
    print translate.translate(test8).lmod

if __name__ == '__main__':
    main()
