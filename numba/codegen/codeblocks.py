# -*- coding: utf-8 -*-

"""
Control flow basic blocks.
"""

from __future__ import print_function, division, absolute_import

from numba.control_flow import flow
from numba.control_flow import basicblocks

import llvm.core

#----------------------------------------------------------------------------
# CFG basic blocks
#----------------------------------------------------------------------------

class CodeBlock(basicblocks.BasicBlock):
    """
    Basic block abstraction for LLVM. Build by the same algorithm as the
    CFG builder, and builds up the same graph with jumps and labels.
    """

    # TODO: Make this generate custom Labels and Branches

    def __init__(self, builder, lfunc, id, label, pos):
        super(CodeBlock, self).__init__(id, label, pos)
        self.builder = builder
        self.lfunc = lfunc
        self.llvm_block = self.lfunc.append_basic_block(label)

#----------------------------------------------------------------------------
# Codegen CFG Flow
#----------------------------------------------------------------------------

class CodeFlow(flow.Flow):

    BasicBlock = CodeBlock

#----------------------------------------------------------------------------
# CFG Lowering
#----------------------------------------------------------------------------

def lower_cfg(flow, builder):
    """
    Connect LLVM basic blocks.
    """
    for block in flow.blocks:
        is_terminated = block.llvm_block.instructions[-1].is_terminator
        if len(block.children) == 1 and not is_terminated:
            builder.position_at_end(block.llvm_block)
            childblock, = block.children
            builder.branch(childblock)
        else:
            assert is_terminated, block
