# -*- coding: utf-8 -*-

"""
Visualize an AST.
"""

from __future__ import print_function, division, absolute_import

import os
import ast
import textwrap
from itertools import chain, imap, ifilter

from numba.viz.graphviz import render

# ______________________________________________________________________
# Utilities

is_ast = lambda node: (isinstance(node, (ast.AST, list)) and not
                       isinstance(node, ast.expr_context))

class NonASTConstant(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return repr(self.value)

def make_list(node):
    if isinstance(node, list):
        return node
    elif isinstance(node, ast.AST):
        return [node]
    else:
        return [NonASTConstant(node)]

def nodes(node):
    return [getattr(node, attr) for attr in node._fields]

def fields(node):
    return zip(node._fields, nodes(node))

# ______________________________________________________________________
# Adaptor

class ASTGraphAdaptor(object):

    def children(self, node):
        return list(chain(*imap(make_list, ifilter(is_ast, nodes(node)))))

# ______________________________________________________________________
# Renderer

def strval(val):
    if isinstance(val, ast.expr_context):
        return type(val).__name__ # Load, Store, Param
    else:
        return str(val)

class ASTGraphRenderer(object):

    def render(self, node):
        args = ", ".join(
            '%s=%s' % (attr, strval(child)) for attr, child in fields(node)
                                                if not is_ast(child))
        return "%s(%s)" % (type(node).__name__, args)

    def render_edge(self, source, dest):
        # See which attribute of the source node matches the destination node
        for attr_name, attr in fields(source):
            if attr is dest or (isinstance(attr, list) and dest in attr):
                # node.attr == dst_node or dest_node in node.attr
                return attr_name

# ______________________________________________________________________
# Entry Point

def render_ast(ast, output_file):
    render([ast], output_file, ASTGraphAdaptor(), ASTGraphRenderer())

# ______________________________________________________________________
# Test

if __name__ == '__main__':
    source = textwrap.dedent("""
        def func(a, b):
            for i in range(10):
                if i < 5:
                    print "hello"
    """)
    mod = ast.parse(source)
    print(mod)
    render_ast(mod, os.path.expanduser("~/ast.dot"))