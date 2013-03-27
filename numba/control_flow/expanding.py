# -*- coding: utf-8 -*-

"""
Expands control flow into basic blocks.

Used to:

    * Build CFG
    * Lower into IR with expanded control flow (jumps and labels)
    * Generate code
"""

from __future__ import print_function, division, absolute_import

import ast

from numba import visitors
from numba import symtab
from numba import nodes
from numba.control_flow.cfstats import *


class ControlFlowExpander(visitors.NumbaTransformer):
    """
    Expand control flow into a CFG. This can also be used to lower into
    IR with expanded control flow and generally code generation.

    The CFG must be build in topological dominator tree order, e.g. the
    'if' condition block must precede the clauses and the clauses must
    precede the exit.
    """

    function_level = 0

    def __init__(self, env, func, ast, cfflow):
        super(ControlFlowExpander, self).__init__(env.context, func, ast,
                                                  env)
        self.flow = cfflow

    def visit(self, node):
        if hasattr(node, 'lineno'):
            self.mark_position(node)

        return super(ControlFlowExpander, self).visit(node)

    def visit_FunctionDef(self, node):
        if self.function_level:
            return node

        self.function_level += 1

        self.visitlist(node.decorator_list)

        self.flow.nextblock(node, 'entry')
        self.mark_position(node)

        # Function body block
        node.body_block = self.flow.nextblock(node, 'function_body')
        for arg in node.args.args:
            if hasattr(arg, 'id') and hasattr(arg, 'ctx'):
                self.visit_Name(arg)
            else:
                self.visit_arg(arg, node.lineno, 0)

        self.visitlist(node.body)
        self.function_level -= 1

        # Exit point
        if self.flow.block:
            self.flow.block.add_child(self.flow.exit_point)

        self.flow.add_floating(self.flow.exit_point)

        return node

    def mark_assignment(self, lhs, rhs=None, assignment=None, **kwds):
        pass

    def mark_position(self, node):
        """Mark position if DOT output is enabled."""

    def visit_Suite(self, node):
        """
        Visit suite. Delete unreachable statements.
        """
        if self.flow.block:
            for i, stat in enumerate(node.body):
                node.body[i] = self.visit(stat)
                if not self.flow.block:
                    # stat.is_terminator = True
                    break

        return node

    def visit_If(self, node):
        with self.flow.float(node, 'exit_if') as node.exit_block:
            # Condition
            node.cond_block = self.flow.nextblock(node.test, 'if_cond')
            node.test = self.visit(node.test)

            self.handle_body(node, node.exit_block)
            self.handle_else_clause(node, node.cond_block, node.exit_block)

        return node

    def handle_body(self, node, exit_block):
        # If Body, child of condition block
        node.if_block = self.flow.nextblock(node.body[0], 'body')
        self.visitlist(node.body)

        if self.flow.block:
            self.flow.block.add_child(exit_block)

    def handle_else_clause(self, node, cond_block, exit_block):
        if node.orelse:
            # Else clause, child of condition block
            node.else_block = self.flow.nextblock(node.orelse[0],
                                                  'else_body', cond_block)
            self.visitlist(node.orelse)

            if self.flow.block:
                self.flow.block.add_child(exit_block)
        else:
            # No else clause, the exit block is a child of the condition
            cond_block.add_child(exit_block)
            node.else_block = None

    def finalize_loop(self, node, cond_block, exit_block):
        if self.flow.block:
            # Add back-edge
            self.flow.block.add_child(cond_block)

        self.handle_else_clause(node, cond_block, exit_block)

    def visit_While(self, node):
        with self.flow.float(node, 'exit_while') as node.exit_block:
            # Condition block
            node.cond_block = self.flow.nextblock(node.test, 'while_condition')
            self.flow.loops.append(LoopDescr(node.exit_block, node.cond_block))
            node.test = self.visit(node.test)

            # Body
            node.while_block = self.flow.nextblock(node.body[0], "while_body")
            self.visitlist(node.body)
            self.flow.loops.pop()

            self.finalize_loop(node, node.cond_block, node.exit_block)

        return node

    def visit_For(self, node):
        # Evaluate iterator in previous block
        node.iter = self.visit(node.iter)

        # Start condition block
        node.cond_block = self.flow.nextblock(node.iter, 'for_condition')

        with self.flow.float(node, 'exit_for') as node.exit_block:
            # Body
            node.for_block = self.flow.nextblock(node.body[0], 'for_body')
            self.flow.loops.append(LoopDescr(node.exit_block, node.cond_block))

            self.visitlist(node.body)
            self.flow.loops.pop()

            self.finalize_loop(node, node.cond_block, node.exit_block)

        return node

    def visit_With(self, node):
        node.context_expr = self.visit(node.context_expr)
        if node.optional_vars:
            # TODO: Mark these as assignments!
            node.optional_vars = self.visit(node.optional_vars)

        # TODO: Build CFG for with blocks

        self.visitlist(node.body)
        return node

    def visit_Raise(self, node):
        self.visitchildren(node)
        if self.flow.exceptions:
            self.flow.block.add_child(self.flow.exceptions[-1].entry_point)

        self.flow.block = None
        return node

    def visit_Return(self, node):
        self.visitchildren(node)

        for exception in self.flow.exceptions[::-1]:
            if exception.finally_enter:
                self.flow.block.add_child(exception.finally_enter)
                if exception.finally_exit:
                    exception.finally_exit.add_child(self.flow.exit_point)
                break
        else:
            if self.flow.block:
                self.flow.block.add_child(self.flow.exit_point)

        self.flow.block = None
        return node

    def visit_Break(self, node):
        if not self.flow.loops:
            return node

        loop = self.flow.loops[-1]
        for exception in loop.exceptions[::-1]:
            if exception.finally_enter:
                self.flow.block.add_child(exception.finally_enter)
                if exception.finally_exit:
                    exception.finally_exit.add_child(loop.next_block)
                break
        else:
            self.flow.block.add_child(loop.next_block)

        self.flow.block = None
        return node

    def visit_Continue(self, node):
        if not self.flow.loops:
            return node

        loop = self.flow.loops[-1]
        for exception in loop.exceptions[::-1]:
            if exception.finally_enter:
                self.flow.block.add_child(exception.finally_enter)
                if exception.finally_exit:
                    exception.finally_exit.add_child(loop.loop_block)
                break
        else:
            self.flow.block.add_child(loop.loop_block)

        self.flow.block = None
        return node
