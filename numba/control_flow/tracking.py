# -*- coding: utf-8 -*-

"""
Control flow tracking.
"""

from __future__ import print_function, division, absolute_import

from numba import visitors

#----------------------------------------------------------------------------
# Control Flow Tracking
#----------------------------------------------------------------------------

class BlockTracker(visitors.NumbaTransformer):
    """
    Track basic blocks in 'self.block' for an AST annotated with a CFG.

    Alternative 1)

        Use nodes with basic blocks as children which contain their wrapping
        statement, e.g.:

            For(body=Block(...)).

        Con: This would mean we have to differentiate between expression and
        statement blocks.

    Alternative 2)

        Use nodes with basic blocks as children, which do NOT contain their
        wrapping statement, but rather are ordered carefully, e.g.:

            For(body_block=Block(), body=[...])

        Con: We have to mutate our AST.

    However, both alternatives would eliminate this pass.
    """

    function_level = 0

    block = None

    def __init__(self, env, func, ast, cfflow):
        super(BlockTracker, self).__init__(env.context, func, ast, env)
        self.flow = cfflow

    def visit_FunctionDef(self, node):
        self.block = self.flow.blocks[0]
        self.visitchildren(node)
        return node

    def visit_block_and_body(self, block, node):
        self.block = block
        return self.visit(node)

    def visit_attrs(self, node, block_attr, body_attr):
        block = getattr(node, block_attr)
        body = getattr(node, body_attr)
        new_body = self.visit_block_and_body(block, body)
        setattr(node, body_attr, new_body)

    def visit_If(self, node):
        self.visit_attrs(node, 'cond_block', 'test')
        self.visit_attrs(node, 'if_block', 'body')
        self.visit_attrs(node, 'else_block', 'orelse')
        self.block = node.exit_block
        return node

    def visit_While(self, node):
        self.visit_attrs(node, 'cond_block', 'test')
        self.visit_attrs(node, 'while_block', 'body')
        self.visit_attrs(node, 'else_block', 'orelse')
        self.block = node.exit_block
        return node

    def visit_For(self, node):
        self.visit_attrs(node, 'cond_block', 'target')
        self.visit_attrs(node, 'for_block', 'body')
        self.visit_attrs(node, 'else_block', 'orelse')
        self.block = node.exit_block
        return node

    def visit_Return(self, node):
        self.visitchildren(node)
        self.block = None
        return node

    def visit_Break(self, node):
        self.block = None
        return node

    def visit_Continue(self, node):
        self.block = None
        return node

    def visit_Raise(self, node):
        self.visitchildren(node)
        self.block = None
        return node

    def visit_With(self, node):
        # TODO: track
        self.visitchildren(node)
        return node
