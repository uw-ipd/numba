# -*- coding: utf-8 -*-

"""
Control flow basic blocks.
"""

from __future__ import print_function, division, absolute_import

import os
import functools

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

    def __init__(self, lfunc, id, label, pos):
        super(CodeBlock, self).__init__(id, label, pos)
        self.lfunc = lfunc
        self.llvm_block = self.lfunc.append_basic_block(label)

        self.phi_defs = []
        self.live_defs = {}
        self.live_promotions = set()

#----------------------------------------------------------------------------
# Codegen CFG Flow
#----------------------------------------------------------------------------

class CodeFlow(flow.Flow):

    def __init__(self, env):
        self.BasicBlock = functools.partial(CodeBlock, env.crnt.lfunc)
        self.builder = None
        super(CodeFlow, self).__init__(env)

    def get_block(self):
        return self._block

    def set_block(self, block):
        self._block = block
        if self.builder:
            self.builder.position_at_end(block.llvm_block)

    block = property(get_block, set_block)

#----------------------------------------------------------------------------
# CFG Lowering
#----------------------------------------------------------------------------

def is_terminated(block):
    instructions = block.llvm_block.instructions
    return instructions and instructions[-1].is_terminator

def lower_cfg(flow, builder, lfunc):
    """
    Connect LLVM basic blocks.
    """
    from numba.viz import cfgviz
    cfgviz.render_cfg(flow, os.path.expanduser("~/cfg.dot"))

    flow.blocks.insert(0, flow.entry_point)
    flow.blocks.append(flow.exit_point)

    for block in flow.blocks:
        print(block, block.children, is_terminated(block))
        if len(block.children) == 1 and not is_terminated(block):
            builder.position_at_end(block.llvm_block)
            childblock, = block.children
            builder.branch(childblock.llvm_block)

    print(lfunc)

    for block in flow.blocks:
        assert is_terminated(block), (block, block.children)
