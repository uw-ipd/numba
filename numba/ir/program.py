# -*- coding: utf-8 -*-

"""
Flow graph and operation for programs.
"""

from __future__ import print_function, division, absolute_import

class FunctionDef(object):

    def __init__(self, ast, func_env, flow):
        self.func_env = func_env
        self.ast = ast

        # Some AST compatibility...
        self.name = ast.name
        self.args = ast.args
        self._fields = ['blocks']
        self.blocks = flow.blocks

        self.body = self.blocks
        self.decorator_list = ast.decorator_list
