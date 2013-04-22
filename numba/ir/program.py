# -*- coding: utf-8 -*-

"""
Flow graph and operation for programs.
"""

from __future__ import print_function, division, absolute_import

class FunctionGraph(object):

    def __init__(self, ast, func_env, flow):
        self.func_env = func_env
        self.ast = ast

        # Some AST compatibility...
        self.args = ast.args
        self._fields = ['blocks']
        self.blocks = flow.blocks


class Block(object):

    def __init__(self, func_graph, predecessors, successors):
        self.parent = func_graph
        self.predecessors
        self.successors = successors

