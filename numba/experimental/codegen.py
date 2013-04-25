from llvmpy.api import llvm
from . import llvm_passes, mangling, llvm_const
from .llvm_utils import (llvm_context, verify_module, verify_function,
                         get_or_insert_func, make_basic_block)
from . import llvm_types

def inline_all_abstract(lmod):
    "Inline all abstract API calls in the module"
    info = llvm.InlineFunctionInfo.new()
    funclist = []
    for lfunc in lmod.list_functions():
        if lfunc.getName().startswith('numba.abstract.'):
            funclist.append(lfunc)
            for use in lfunc.list_use():
                if use._downcast(llvm.Instruction).getOpcodeName() == 'call':
                    callinst = use._downcast(llvm.CallInst)
                    if callinst.getCalledFunction() == lfunc:
                        llvm.InlineFunction(callinst, info)
    for lfunc in funclist:
        if not lfunc.list_use():
            lfunc.eraseFromParent()

def error_handling(lmod):
    "Add error handling to all Python API calls"
    builder = llvm.IRBuilder.new(llvm_context)
    for lfunc in lmod.list_functions():
        if not lfunc.isDeclaration():
            # find exit node
            terminator = _find_return(lfunc)
            # find all call that returns an object
            calls = []
            for bb in lfunc.getBasicBlockList():
                ilist = bb.getInstList()
                for inst in ilist:
                    if inst.getOpcodeName() == 'call':
                        callinst = inst._downcast(llvm.CallInst)
                        func = callinst.getCalledFunction()
                        name = func.getName()
                        if (name.startswith('Py') and
                            func.getReturnType() == llvm_types.unknown_ptr):
                            calls.append(callinst)
            for callinst in calls:
                builder.SetInsertPoint(callinst.getNextNode())
                icmppred = llvm.CmpInst.Predicate
                null = llvm_const.null(callinst.getType())
                cmp = builder.CreateICmp(icmppred.ICMP_EQ, callinst, null)
                if callinst.getCalledFunction().getName() == 'PyIter_Next':
                    # also check if exception has occurred
                    errchk = get_or_insert_func(lmod, 'PyErr_Occurred',
                                                llvm_types.unknown_ptr, [])
                    errchkres = builder.CreateCall(errchk, ())
                    haserr = builder.CreateICmp(icmppred.ICMP_NE, errchkres,
                                                null)
                    cmp = builder.CreateAnd(cmp, haserr)
                terminst = llvm.SplitBlockAndInsertIfThen(cmp, False, None)
                bbnew = terminst.getParent()
                terminst.eraseFromParent()
                builder.SetInsertPoint(bbnew)
                retinst = builder.CreateRet(null)


            verify_function(lfunc)
    verify_module(lmod)

def _find_return(lfunc):
    blocks = lfunc.getBasicBlockList()
    for bb in blocks:
        terminator = bb.getTerminator()
        if terminator.getOpcodeName() == 'ret':
            terminator = terminator._downcast(llvm.ReturnInst)
            break
    else:
        raise Exception('Missing return')
    return terminator

def lower(lmod):
    "Define all object calls"
    for lfunc in lmod.list_functions():
        if lfunc.isDeclaration():
            fname = lfunc.getName()
            if fname.startswith('numba.abstract.pack.'):
                define_pack(lmod, lfunc, _prepare_builder(lmod, lfunc))
            elif fname.startswith('numba.'):
                define_abstract_func(lmod, lfunc)

    verify_module(lmod)

_func_defs = {}

def define(name, *argtys):
    def _define(fn):
        _func_defs[mangling.mangle(name, argtys)] = fn
    return _define

def get_definition(lfunc):
    name = lfunc.getName()
    return _func_defs[name]

def define_abstract(name, *argtys):
    return define('numba.abstract.' + name, *argtys)

def define_abstract_func(lmod, lfunc):
    builder = _prepare_builder(lmod, lfunc)
    define = get_definition(lfunc)
    define(lmod, lfunc, builder)

def _prepare_builder(lmod, lfunc):
    builder = llvm.IRBuilder.new(llvm_context)
    bb = make_basic_block(lfunc)
    builder.SetInsertPoint(bb)
    return builder

#
# Definitions
#

def define_pack(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    packfn = get_or_insert_func(lmod, "PyTuple_Pack", llvm_types.unknown_ptr,
                                [llvm_types.py_ssize_t], vararg=True)
    n = llvm_const.integer(llvm_types.py_ssize_t, len(args))
    res = builder.CreateCall(packfn, [n] + args)
    builder.CreateRet(res)

@define_abstract('binop.add', llvm_types.unknown_ptr, llvm_types.unknown_ptr)
def define_add(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, "PyNumber_Add",
                              llvm_types.unknown_ptr, args)
    res = builder.CreateCall(func, args)
    builder.CreateRet(res)

@define_abstract('binop.sub', llvm_types.unknown_ptr, llvm_types.unknown_ptr)
def define_add(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, "PyNumber_Subtract",
                              llvm_types.unknown_ptr, args)
    res = builder.CreateCall(func, args)
    builder.CreateRet(res)

@define_abstract('binop.mul', llvm_types.unknown_ptr, llvm_types.unknown_ptr)
def define_multiply(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, "PyNumber_Multiply",
                              llvm_types.unknown_ptr, args)
    res = builder.CreateCall(func, args)
    builder.CreateRet(res)

@define_abstract('binop.div', llvm_types.unknown_ptr, llvm_types.unknown_ptr)
def define_multiply(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, "PyNumber_Divide",
                              llvm_types.unknown_ptr, args)
    res = builder.CreateCall(func, args)
    builder.CreateRet(res)


pyapi_cmp = {
    'Py_LT': 0,
    'Py_LE': 1,
    'Py_EQ': 2,
    'Py_NE': 3,
    'Py_GT': 4,
    'Py_GE': 5,
}

@define_abstract('cmpop.gt', llvm_types.unknown_ptr, llvm_types.unknown_ptr)
def define_multiply(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    cmpargs = args + [llvm_const.integer(llvm_types.i32, pyapi_cmp['Py_GT'])]
    func = get_or_insert_func(lmod, "PyObject_RichCompareBool",
                              llvm_types.i32, cmpargs)
    res = builder.CreateCall(func, cmpargs)
    # FIXME: ignoring error condition
    icmpop = llvm.CmpInst.Predicate.ICMP_NE
    zero = llvm_const.null(res.getType())
    res_as_pred = builder.CreateICmp(icmpop, res, zero)
    builder.CreateRet(res_as_pred)


@define_abstract('const', llvm_types.i32)
def define_const_i32(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, "PyInt_FromLong", llvm_types.unknown_ptr,
                              args)
    res = builder.CreateCall(func, args)
    builder.CreateRet(res)

@define_abstract('call', *[llvm_types.unknown_ptr] * 3)
def define_call(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, "PyObject_Call", llvm_types.unknown_ptr,
                              [llvm_types.unknown_ptr] * 3)
    res = builder.CreateCall(func, args)
    builder.CreateRet(res)

@define_abstract('get_iter', llvm_types.unknown_ptr)
def define_call(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, "PyObject_GetIter", llvm_types.unknown_ptr,
                              [llvm_types.unknown_ptr])
    res = builder.CreateCall(func, args)
    builder.CreateRet(res)

@define_abstract('iter.next', llvm_types.unknown_ptr)
def define_call(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    args = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, "PyIter_Next", llvm_types.unknown_ptr,
                              [llvm_types.unknown_ptr])
    res = builder.CreateCall(func, args)
    builder.CreateRet(res)

@define_abstract('iter.empty', llvm_types.unknown_ptr)
def define_call(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    (iterator,) = lfunc.getArgumentList()
    null = llvm_const.null(iterator.getType())
    icmpop = llvm.CmpInst.Predicate.ICMP_EQ
    res = builder.CreateICmp(icmpop, iterator, null)
    builder.CreateRet(res)

@define_abstract('load_attr', llvm_types.unknown_ptr, llvm_types.string)
def define_load_atr(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    obj, attr = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, 'PyObject_GetAttrString',
                              llvm_types.unknown_ptr, [obj, attr])
    res = builder.CreateCall(func, [obj, attr])
    builder.CreateRet(res)

@define_abstract('binary_subscr', llvm_types.unknown_ptr, llvm_types.unknown_ptr)
def define_load_atr(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    obj, key = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, 'PyObject_GetItem',
                              llvm_types.unknown_ptr, [obj, key])
    res = builder.CreateCall(func, [obj, key])
    builder.CreateRet(res)

@define_abstract('store_subscr', *[llvm_types.unknown_ptr] * 3)
def define_load_atr(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    obj, key, val = lfunc.getArgumentList()
    func = get_or_insert_func(lmod, 'PyObject_SetItem',
                              llvm_types.i32, [obj, key, val])
    res = builder.CreateCall(func, [obj, key, val])
    builder.CreateRet(llvm_const.null(llvm_types.unknown_ptr))

@define_abstract('decref', llvm_types.unknown_ptr)
def define_decref(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    bbexit = make_basic_block(lfunc)
    bbwork = make_basic_block(lfunc)
    args = lfunc.getArgumentList()
    ptr = args[0]
    icmpop = llvm.CmpInst.Predicate.ICMP_EQ
    null = llvm_const.null(ptr.getType())
    pred = builder.CreateICmp(icmpop, ptr, null)
    builder.CreateCondBr(pred, bbexit, bbwork)

    builder.SetInsertPoint(bbwork)
    func = get_or_insert_func(lmod, "Py_DecRef", llvm_types.void, args)
    builder.CreateCall(func, args)
    builder.CreateBr(bbexit)

    builder.SetInsertPoint(bbexit)
    builder.CreateRet(null)

@define_abstract('incref', llvm_types.unknown_ptr)
def define_incref(lmod, lfunc, builder):
    lfunc.addFnAttr(llvm.Attributes.AttrVal.AlwaysInline)
    lfunc.setLinkage(llvm.GlobalValue.LinkageTypes.InternalLinkage)
    bbexit = make_basic_block(lfunc)
    bbwork = make_basic_block(lfunc)
    args = lfunc.getArgumentList()
    ptr = args[0]
    icmpop = llvm.CmpInst.Predicate.ICMP_EQ
    null = llvm_const.null(ptr.getType())
    pred = builder.CreateICmp(icmpop, ptr, null)
    builder.CreateCondBr(pred, bbexit, bbwork)

    builder.SetInsertPoint(bbwork)
    func = get_or_insert_func(lmod, "Py_IncRef", llvm_types.void, args)
    builder.CreateCall(func, args)
    builder.CreateBr(bbexit)

    builder.SetInsertPoint(bbexit)
    builder.CreateRet(null)



