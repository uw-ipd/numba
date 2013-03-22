# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ast

from numba.nodes import *

#----------------------------------------------------------------------------
# Control Flow Warning Nodes
#----------------------------------------------------------------------------

class MaybeUnusedNode(Node):
    """
    Wraps an ast.Name() to indicate that the result may be unused.
    """

    _fields = ["name_node"]

    def __init__(self, name_node):
        self.name_node = name_node


#----------------------------------------------------------------------------
# Control Flow Nodes
#----------------------------------------------------------------------------

class For(ast.For):

    _fields = (
        'target',
        'iter',
        'body',
        'incr',
        'orelse',
    )

    def __init__(self, target, iter, body, incr, orelse):
        self.target = target
        self.iter = iter
        self.body = body
        self.incr = incr
        self.orelse = orelse


class While(ast.While):

    _fields = (
        'test',
        'body',
        'incr', # Set to [stmt] if this was a for loop rewritten to a while
        'orelse',
    )

    def __init__(self, test, body, incr, orelse):
        self.test = test
        self.body = body
        self.incr = incr
        self.orelse = orelse


#----------------------------------------------------------------------------
# Utilities
#----------------------------------------------------------------------------

def if_else(op, cond_left, cond_right, lhs, rhs):
    "Implements 'lhs if cond_left <op> cond_right else rhs'"
    test = ast.Compare(left=cond_left, ops=[op],
                       comparators=[cond_right])
    test.right = cond_right
    test = typednode(test, bool_)

    return ast.If(test=test, body=[lhs], orelse=[rhs] if rhs else [])
