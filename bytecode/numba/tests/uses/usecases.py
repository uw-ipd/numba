
def usecase1(a, b, c):
    return a + b * c / 4

def usecase2(a, b, c):
    if a > b:
        c += a
    else:
        c += b
    return c

def usecase3(a, b):
    for i in range(1, a):
        b += i
    return b

def usecase4(a, b, c):
    for i in range(1, a):
        c += i
        for j in range(b):
            c += j
    return c

K1 = 123
def usecase5(a, b, c):
    return a + b * c / K1

def usecase6(a, b, c):
    return usecase5(a, b, c)

def usecase7(a, b, c):
    if a > b:
        return usecase7(a, b + 1, c)
    else:
        return c

def usecase8(a, b, c):
    for i in range(a.shape[0]):
        a[i] += b[i] + c[i]
    return a

def usecase9(a, b, c):
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            for k in range(c.shape[0]):
                a[i] += b[j] + c[k]
    return a
