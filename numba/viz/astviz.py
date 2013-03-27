# -*- coding: utf-8 -*-

"""
Visualize an AST.
"""

from __future__ import print_function, division, absolute_import

import os
import ast
import textwrap
from itertools import chain

from numba.viz.graphviz import render

# ______________________________________________________________________
# Adaptor

is_ast = lambda node: isinstance(node, (ast.AST, list))

class ASTGraphAdaptor(object):

    def children(self, node):
        if not is_ast(node):
            return []
        nodes = [getattr(node, attr) for attr in node._fields]
        return list(chain(*[n if isinstance(n, list) else [n] for n in nodes]))

# ______________________________________________________________________
# Renderer

class ASTGraphRenderer(object):

    children = ASTGraphAdaptor().children
    fields = lambda self, node: zip(node._fields, self.children(node))

    def render(self, node):
        if not is_ast(node):
            return str(node)

        fields = self.fields(node)
        args = ", ".join('%s=%s' % (attr, child) for attr, child in fields
                             if not isinstance(child, ast.AST))
        return "%s(%s)" % (type(node).__name__, args)

    def render_edge(self, source, dest):
        for attr_name, attr in self.fields(source):
            if attr is dest:
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