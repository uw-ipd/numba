
def mangle(name, argtys):
    if argtys:
        mangled = '%s.%s' % (name, '.'.join(map(mangle_type, argtys)))
    else:
        mangled = name
    return mangled

def mangle_type(ty):
    if ty.isStructTy():
        return ty.getStructName()
    if ty.isPointerTy():
        return "%s_p" % mangle_type(ty.getPointerElementType())
    if ty.isArrayTy():
        elemty = ty.getArrayElementType()
        elemct = ty.getArrayNumElements()
        return "%dx%s" % (elemct, elemty)
    else:
        return str(ty)

