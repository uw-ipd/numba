import ctypes
from llvmpy.api import llvm
from .global_mappings import builtins_table
from . import llvm_types

def make_engine(lmod, opt=2):
    llvm.InitializeNativeTarget()
    engine = llvm.EngineBuilder.new(lmod).setOptLevel(opt).create()
    assert engine
    return engine

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


class ExecutionContext(object):
    def __init__(self, lmod, opt=2):
        self.engine = make_engine(lmod, opt=opt)
        self._preserve_mapping = add_global_mapping(self.engine, lmod)

    def prepare(self, lfunc):
        wrapped = make_callable(self.engine, lfunc)
        return wrapped
