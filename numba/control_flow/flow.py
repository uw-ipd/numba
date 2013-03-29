# -*- coding: utf-8 -*-

"""
Control flow blocks. These constitute the CFG, which is separate from the
AST or any code blocks in the AST (see cfnodes for that).
"""

from __future__ import print_function, division, absolute_import

import re

from numba import error, symtab

from numba import *
from numba.control_flow import reaching
from numba.control_flow import cfablocks
from numba.control_flow.cfstats import *
from numba.control_flow.debug import *

#----------------------------------------------------------------------------
# Content Manager for Basic Blocks
#----------------------------------------------------------------------------

class FloatingBlockContext(object):
    """
    Create a floating block, which can be used to connect other blocks to,
    before commiting it to the CFG. Using this will ensure a topological
    ordering of the CFG blocks.

        with flow.float(pos, "my_block"):
            ...
    """

    def __init__(self, flow, block):
        self.flow = flow
        self.block = block

    def __enter__(self):
        return self.block

    def __exit__(self, *args):
        self.flow.add_floating(self.block)

        if self.block.parents:
            self.flow.block = self.block
        else:
            self.flow.block = None

#----------------------------------------------------------------------------
# Control Flow Graph Builder
#----------------------------------------------------------------------------

class Flow(object):
    """
    Control-flow graph.

       entry_point ControlBlock entry point for this graph
       exit_point  ControlBlock normal exit point
       block       ControlBlock current block
       blocks      set    children nodes
       entries     set    tracked entries
       loops       list   stack for loop descriptors
       exceptions  list   stack for exception descriptors

    """

    # BasicBlock class
    BasicBlock = None

    def __init__(self, env):
        self.env = env

        self.blocks = []
        self.entries = set()
        self.loops = []
        self.exceptions = []

        self.entry_point = self.BasicBlock(-1, label='entry', pos=None)
        self.exit_point = self.BasicBlock(0, label='exit', pos=None)
        self.block = self.entry_point

    def newblock(self, pos, label, parent=None):
        """
        Create floating block linked to `parent` if given.
        Does NOT set the current block to the new block.
        """
        id = len(self.blocks)
        block = self.BasicBlock(id, pos=pos, label=label)
        self.blocks.append(block)
        if parent:
            parent.add_child(block)

        return block

    def nextblock(self, pos, label, parent=None):
        """
        Create child block linked to current or `parent` if given.
        Sets the current block to the new block.
        """
        block = self.newblock(pos, label, parent)
        if not parent and self.block:
            self.block.add_child(block)

        self.block = block
        return block

    def add_floating(self, block):
        """
        Add an floating block after visiting the body. Add only if the block
        is parented.
        """
        if block.parents:
            block.id = len(self.blocks)
            self.blocks.append(block)

    def float(self, pos, label, parent=None):
        block = self.newblock(pos, label, parent)
        self.blocks.pop()
        return FloatingBlockContext(self, block)

#----------------------------------------------------------------------------
# Control Flow Graph Builder for CFA
#----------------------------------------------------------------------------

class CFGFlow(Flow):
    """
    Build a control flow graph containing abstract control flow statements:

        * Assignments
        * References
        * Deletions
    """

    BasicBlock = cfablocks.ControlBlock

    def is_listcomp_var(self, name):
        return re.match(r"_\[\d+\]", name)

    def is_tracked(self, entry):
        return (# entry.renameable and not
                entry.name not in self.env.translation.crnt.locals and not
                self.is_listcomp_var(entry.name))

    def mark_position(self, node):
        """Mark position, will be used to draw graph nodes."""

    def mark_assignment(self, lhs, rhs, entry, assignment, warn_unused=True):
        if self.block:
            if not self.is_tracked(entry):
                return
            assignment = NameAssignment(lhs, rhs, entry, assignment,
                                        warn_unused=warn_unused)
            self.block.stats.append(assignment)
            self.block.gen[entry] = assignment
            self.entries.add(entry)
            return assignment

    def mark_argument(self, lhs, rhs, entry):
        if self.block and self.is_tracked(entry):
            assignment = Argument(lhs, rhs, entry)
            self.block.stats.append(assignment)
            self.block.gen[entry] = assignment
            self.entries.add(entry)

    def mark_deletion(self, node, entry):
        if self.block and self.is_tracked(entry):
            assignment = NameDeletion(node, entry)
            self.block.stats.append(assignment)
            self.block.gen[entry] = Uninitialized
            self.entries.add(entry)

    def mark_reference(self, node, entry):
        if self.block and self.is_tracked(entry):
            self.block.stats.append(NameReference(node, entry))
            # Local variable is definitely bound after this reference
            if not reaching.allow_null(node):
                self.block.bound.add(entry)
            self.entries.add(entry)
