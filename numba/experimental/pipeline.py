'''
Defines the default compilation pipeline.
'''
import dis
from collections import namedtuple
from . import translate, llvm_passes, codegen

CompiledInfo = namedtuple('CompiledInfo', ['func', 'lmod', 'lfunc'])

def compile(fn, debug=False):
    '''
    Compile a function.
    '''
    # translate from bytecode
    if debug:
        dis.disassemble(fn.func_code)
    trans = translate.translate(fn)
    lmod = trans.lmod
    lfunc = trans.lfunc
    if debug:
        print(lmod)
    # lowering
    translate.remove_redundant_refct(lmod)
    codegen.lower(lmod)
    codegen.inline_all_abstract(lmod)
    codegen.error_handling(lmod)

    # optimize
    llvm_passes.make_pm().run(trans.lmod)

    ci = CompiledInfo(func=fn, lmod=lmod, lfunc=lfunc)
    return ci


