# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

from numba.experimental import llvm_passes
from numba.control_flow import flowy
print(flowy)
from numba.control_flow.flowy import Opcode

simple_ops = [
    # Need a data dependence graph to optimize these
    Opcode("load", read=True),
    Opcode("store", read=False),

    # Some opcodes we try to optimize
    Opcode("tuple_new", canfold=True),
    Opcode("int_eq", read=False, sideeffects=False, canfold=True),
    Opcode("int_mul", read=False, sideeffects=False, canfold=True),

    # Branches
    Opcode("cbranch"),
]

opdict = dict((opcode.op, opcode) for opcode in simple_ops)

class OpContext(flowy.OperationContext):

    def is_terminator(self, operation):
        return operation.opcode.op == "cbranch"

    def is_conditional_branch(self, operation):
        return operation.opcode.op == "cbranch"

    def get_condition(self, conditional_branch):
        return conditional_branch.args[0]

    def is_boolean_operation(self, operation):
        return operation.opcode.op in ("int_eq",)

opctx = OpContext()

def make_testprogram():
    g = flowy.FunctionGraph("test_program")
    builder = flowy.OperationBuilder(g)

    # if x: ...
    entry = builder.add_block([], "entry")
    cond_block = builder.add_block([entry], "cond")
    loop_block = builder.add_block([cond_block], "loop")
    exit_block = builder.add_block([cond_block])

    cond_block.add_parents(loop_block)

    c1 = builder.const(1)
    c2 = builder.const(2)

    eq = builder.op(opdict["int_eq"], [c1, c2])
    cbr = builder.op(opdict["cbranch"], [eq])
    op1 = builder.op(opdict["int_mul"], [c1, c2])
    op2 = builder.op(opdict["tuple_new"], [c1, c2, op1])

    entry.append(eq)
    cond_block.append(cbr)
    loop_block.extend([op1, op2])

    return g

def get_passes():
    from llvmpy.api import llvm

    passreg = llvm.PassRegistry.getPassRegistry()

    llvm.initializeCore(passreg)
    llvm.initializeScalarOpts(passreg)
    llvm.initializeVectorization(passreg)
    llvm.initializeIPO(passreg)
    llvm.initializeAnalysis(passreg)
    llvm.initializeIPA(passreg)
    llvm.initializeTransformUtils(passreg)
    llvm.initializeInstCombine(passreg)
    llvm.initializeInstrumentation(passreg)
    llvm.initializeTarget(passreg)

    def _dump_all_passes():
        for name, desc in passreg.enumerate():
            yield name, desc
    return dict(_dump_all_passes())

def LICM():
    g = make_testprogram()
    print(g)
    llvm_mapper = flowy.LLVMMapper(g, opctx)
    lfunc = llvm_mapper.make_llvm_graph()
    print(lfunc)

    llvm_passes.run_function_passses(lfunc, get_passes()) #llvm_passes.PASSES)
    print(lfunc)

if __name__ == "__main__":
    # LICM()
    pass