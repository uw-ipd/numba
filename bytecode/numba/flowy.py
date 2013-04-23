# -*- coding: utf-8 -*-

"""
Flow graph and operation for programs.
"""

from __future__ import print_function, division, absolute_import

import llvm.core
from . import llvm_passes, llvm_types, llvm_utils
from .llvm_utils import llvm_context

class FunctionGraph(object):

    def __init__(self, blocks):
        self.blocks = blocks

class Block(object):

    def __init__(self, func_graph, predecessors, successors):
        self.parent = func_graph
        self.predecessors = predecessors
        self.successors = successors
        self.operations = []

class Opcode(object):
    """
    Opcode used to indicate the type of Operation.

        op: an opcode representation, e.g. the string 'call'
        sideeffects: do operations of this opcode have sideeffects?
        read: do operations of this opcode read memory?
        write: do operations of this opcode write memory?
        canfold: can we constant fold operations of this opcode?
                 (if they have constant arguments)
        exceptions: set of exceptions the operation may raise
    """

    def __init__(self, concrete_opcode, sideeffects=True,
                 read=True, write=True, canfold=True,
                 exceptions=()):
        self.op = concrete_opcode

        self.sideeffects = sideeffects or write
        self.read = read
        self.write = write
        self.canfold = canfold
        self.exceptions = exceptions


class Operation(object):

    def __init__(self, opcode, args, result):
        self.opcode = opcode
        self.args = args
        self.result = result


class Value(object):
    """
    The result of an Operation.
    """

    is_var = False
    is_const = False


class Variable(Value):

    is_var = True

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return "%" + self.name


class Constant(Value):
    "Constant value. Immutable!"

    is_const = True

    def __init__(self, const):
        self._const = const

    @property
    def const(self):
        return self._const

    def __repr__(self):
        return "const(%s)" % self.const

# ______________________________________________________________________

class FunctionGraphContext(object):

    def __init__(self, opctx, constfolder):
        self.opctx = opctx
        self.constfolder = constfolder

    def return_value(self, funcgraph):
        """
        :return: The return value of this function represented by `funcgraph`
        """
        return None

    # ....
    # delegations here

class OperationContext(object):

    def is_pure(self, operation):
        op = operation.opcode
        return not op.sideeffects and op.canfold

    def is_terminator(self, operation):
        raise NotImplementedError("is_terminator")

    def is_return(self, operation):
        raise NotImplementedError("is_return")

    def is_conditional_branch(self, operation):
        raise NotImplementedError("is_conditional_branch")

    def get_conditional_branch(self, operation):
        raise NotImplementedError("get_conditional_branch")

    def opname(self, opcode):
        return str(opcode.op)


class ConstantFolder(object):

    def fold(self, operation):
        """
        Try to fold the operation if all arguments are Constant.

        :return: Constant or Variable

            In case the operation cannot be folded, simply return `operation`
        """
        return operation

# ______________________________________________________________________
# LLVM stuff

llvm_value_t = llvm.StructType.create(llvm_context, "unknown")
unknown_ptr = llvm.PointerType.getUnqual(llvm_value_t)

class LLVMBuilder(object):

    def __init__(self, name, opague_type, argnames):
        self.name = name
        self.opague_type = opague_type
        self.lmod = self.make_module(name)
        self.lfunc = self.make_func(self.lmod, name, opague_type, argnames)
        self.builder = self.make_builder(self.lfunc)

    # __________________________________________________________________

    @classmethod
    def make_module(cls, name):
        mod = llvm.Module.new('module.%s' % name, llvm_context)
        return mod

    @classmethod
    def make_func(cls, lmod, name, opague_type, argnames):
        argtys = [llvm_types.unknown_ptr] * len(argnames)
        lfunc = llvm_utils.get_or_insert_func(
            lmod, name, opague_type, argtys)
        return lfunc

    @classmethod
    def make_builder(cls, lfunc):
        entry = cls.make_block(lfunc, "entry")
        builder = llvm.IRBuilder.new(llvm_context)
        builder.SetInsertPoint(entry)
        return builder

    @classmethod
    def make_block(cls, lfunc, blockname):
        blockname = 'block_%s' % blockname
        return llvm_utils.make_basic_block(lfunc, blockname)

    # __________________________________________________________________

    def delete(self):
        self.lmod = None
        self.lfunc = None
        self.builder = None

    def verify(self):
        llvm_utils.verify_module(self.lmod)

    def add_block(self, blockname):
        return self.make_block(self.lfunc, blockname)

    def set_block(self, dstblock):
        self.builder.SetInsertPoint(dstblock)

    def run_passes(self, passes):
        llvm_passes.run_function_passses(self.lfunc, passes)

    def call_abstract(self, name, *args):
        argtys = [x.getType() for x in args]
        callee = llvm_utils.get_or_insert_func(self.lmod, name,
                                               self.opague_type, argtys)
        return self.builder.CreateCall(callee, args)

    def call_abstract_pred(self, name, *args):
        is_vararg = False
        argtys = [x.getType() for x in args]
        retty = llvm_types.i1
        callee = llvm_utils.get_or_insert_func(self.lmod, name,
                                               retty, argtys)
        return self.builder.CreateCall(callee, args)


class LLVMMapper(object):

    LLVMBuilder = LLVMBuilder

    def __init__(self, funcgraph, opctx):
        self.funcgraph = funcgraph
        self.opctx = opctx
        self.builder = self.LLVMBuilder()

        # Operation -> LLVM Value
        self.llvm_values = {}

        # Block -> llvm block
        self.llvm_blocks = {}

    def llvm_operation(self, operation, llvm_args):
        name = self.opctx.opname(operation.opcode)
        # include operation arity in name
        name = '%s_%d' % (name, len(operation.args))
        return self.builder.call_abstract(name, llvm_args)

    def process_op(self, operation):
        args = [self.llvm_values[arg] for arg in operation.args]
        value = self.llvm_operation(operation, args)
        self.llvm_values[operation] = value

    def make_llvm_graph(self):
        "Populate the LLVM Function with abstract IR"

        # Allocate blocks
        for block in self.funcgraph.blocks:
            self.llvm_blocks[block] = self.builder.add_block(self.builder)

        # Generete abstract IR
        for block in self.funcgraph.blocks:
            for operation in block.operations[:-1]:
                self.process_op(operation)

            if len(block.successors) == 1:
                succ, = block.successors
                self.builder.builder.CreateBr(self.llvm_blocks[succ])
            elif block.operations:
                self.terminate_block(block)

        self.builder.verify()

    def terminate_block(self, block):
        "Terminate a block with conditional branch"
        op = block.operations[-1]

        assert self.opctx.is_terminator(op)
        assert self.opctx.is_conditional_branch(op)
        assert len(block.successors) == 2

        cond = self.opctx.get_conditional_branch(op)
        lcond = self.llvm_values[cond]

        succ1, succ2 = block.successors
        self.builder.builder.CreateCondBr(
            lcond, self.llvm_blocks[succ1], self.llvm_blocks[succ2])
