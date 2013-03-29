# -*- coding: utf-8 -*-

"""
Initial AST validation and normalization.
"""

from __future__ import print_function, division, absolute_import

import ast
from numba import error

class ValidateAST(ast.NodeVisitor):
    "Validate AST"

    #------------------------------------------------------------------------
    # Validation
    #------------------------------------------------------------------------

    def visit_GeneratorExp(self, node):
        raise error.NumbaError(
                node, "Generator comprehensions are not yet supported")

    def visit_SetComp(self, node):
        raise error.NumbaError(
                node, "Set comprehensions are not yet supported")

    def visit_DictComp(self, node):
        raise error.NumbaError(
                node, "Dict comprehensions are not yet supported")

    def visit_Raise(self, node):
        raise error.NumbaError(node, "Raise statement not implemented yet")
