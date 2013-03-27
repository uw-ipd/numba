# -*- coding: utf-8 -*-

"""
Create def/use chains and vice versa.

A definition is:

    * A variable assignment instance ('x = 0')
    * A phi at a control flow merge point

A use is a variable read:

    'x'
"""

from __future__ import print_function, division, absolute_import

import ast

from numba import nodes
from numba import symtab
from numba import visitors
from numba.nodes import cfnodes

#------------------------------------------------------------------------
# Def/use chainer
#------------------------------------------------------------------------

class DefUseChainer(visitors.NumbaTransformer):
    """
    Merge PHIs into AST. This happens after the CFG was build and the
    phis computed.
    """

