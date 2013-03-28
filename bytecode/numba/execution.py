import ctypes
from llvmpy.api import llvm
from llvm import _intrinsic_ids
from .global_mappings import builtins_table
from . import llvm_types, llvm_const
from .llvm_utils import (llvm_context, verify_module, verify_function,
                         get_or_insert_func, make_basic_block)

def make_engine(lmod, opt=2):
    llvm.InitializeNativeTarget()
    engine = llvm.EngineBuilder.new(lmod).setUseMCJIT(True).setOptLevel(opt).create()
    assert engine
    return engine

# FIXME: not used because setjmp does not work
def make_wrapper(lmod, lfunc):
    "Add a wrapper to contain the error handling bits"
    ofnty = lfunc.getFunctionType()
    retty = ofnty.getReturnType()
    argtys = [ofnty.getParamType(i) for i in range(ofnty.getNumParams())]

    wrapper = get_or_insert_func(lmod, "_nb_w_%s" % lfunc.getName(),
                                  retty, argtys)
    assert wrapper.isDeclaration()

    bb = make_basic_block(wrapper)
    bbtry = make_basic_block(wrapper)
    bbcatch = make_basic_block(wrapper)

    builder = llvm.IRBuilder.new(llvm_context)
    builder.SetInsertPoint(bb)

    setjmp = get_or_insert_func(lmod, 'setjmp', llvm_types.i32,
                                [llvm_types.void_ptr])


    ctxsize = llvm_const.integer(llvm_types.i32, 8 * 48) # XXX 384 bytes?
    ctx = builder.CreateAlloca(llvm_types.i8, ctxsize)
    # setjmp does not work; codegen fails
    status = builder.CreateCall(setjmp, [ctx])
    zero = llvm_const.null(llvm_types.i32)
    status = zero

    icmppred = llvm.CmpInst.Predicate
    pred = builder.CreateICmp(icmppred.ICMP_EQ, status, zero)
    builder.CreateCondBr(pred, bbtry, bbcatch)

    # try label
    builder.SetInsertPoint(bbtry)

    res = builder.CreateCall(lfunc, wrapper.getArgumentList())
    builder.CreateRet(res)

    # catch label
    builder.SetInsertPoint(bbcatch)
    builder.CreateRet(llvm_const.null(retty))

    verify_function(wrapper)
    verify_module(lmod)

    return wrapper


def make_callable(engine, lfunc):
    ptr = engine.getPointerToFunction(lfunc)
    args = lfunc.getArgumentList()
    argtys = [arg.getType() for arg in args]
    use_object_call = llvm_types.unknown_ptr in argtys
    functype = ctypes.PYFUNCTYPE if use_object_call else ctypes.CFUNCTYPE
    cargtys = [map_to_ctype(lty) for lty in argtys]
    cretty = map_to_ctype(lfunc.getReturnType())
    prototype = functype(cretty, *cargtys)
    return prototype(ptr)

def map_to_ctype(lty):
    "Map LLVM type to ctype"
    if lty == llvm_types.unknown_ptr:
        return ctypes.py_object
    else:
        raise NotImplementedError(lty)


_global_mapping = dict(("numba.%s" % v, k) for k, v in builtins_table.items())

def add_global_mapping(engine, lmod):
    "Map builtin object into global variables of the module"
    preserving = []
    for gv in lmod.list_globals():
        name = gv.getName()
        try:
            obj = _global_mapping[name]
        except KeyError:
            pass
        else:
            assert not gv.getInitializer()
            addr = ctypes.c_void_p(id(obj))
            preserving.append(addr)
            engine.addGlobalMapping(gv, ctypes.addressof(addr))
    return preserving

def map_global_variables(engine, lmod, globals):
    "Map object id to the module as global mapping"
    preserving = []
    for gv in lmod.list_globals():
        gvname = gv.getName()
        prefix = 'numba.global.'
        if gvname.startswith(prefix):
            assert not gv.getInitializer()
            key = gvname[len(prefix):]
            obj = globals[key]
            addr = ctypes.c_void_p(id(obj))
            engine.addGlobalMapping(gv, ctypes.addressof(addr))
    return preserving

class ExecutionContext(object):
    def __init__(self, lmod, opt=2):
        self.lmod = lmod
        self.engine = make_engine(lmod, opt=opt)
        self._preserve_mapping = add_global_mapping(self.engine, lmod)

    def prepare(self, lfunc, globals={}):
        gv_preserve = map_global_variables(self.engine, self.lmod, globals)
        self._preserve_mapping.extend(gv_preserve)
        wrapped = make_callable(self.engine, lfunc)
        return wrapped

