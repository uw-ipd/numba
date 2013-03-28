# -*- coding: utf-8 -*-

"""
Visualize an AST.
"""

from __future__ import print_function, division, absolute_import

import os
import ast
import textwrap

from numba import void
from numba import pipeline
from numba import environment
from numba.viz.graphviz import render
from numba.control_flow import entrypoints
from numba.control_flow.cfstats import NameAssignment

# ______________________________________________________________________
# Adaptor

class SSAGraphAdaptor(object):
    def children(self, node):
        return [nameref.variable for nameref in node.cf_references]

# ______________________________________________________________________
# Renderer

class SSAGraphRenderer(object):

    def render(self, node):
        if node.renamed_name:
            return node.unmangled_name
        return node.name

    def render_edge(self, source, dest):
        return "use"

# ______________________________________________________________________
# Entry Points

def render_ssa(cfflow, symtab, output_file):
    "Render the SSA graph given the flow.CFGFlow and the symbol table"
    cfstats = [stat for b in cfflow.blocks for stat in b.stats]
    defs = [stat.lhs.variable for stat in cfstats
                                  if isinstance(stat, NameAssignment)]
    nodes = symtab.values() + defs
    render(nodes, output_file, SSAGraphAdaptor(), SSAGraphRenderer())

def render_ssa_from_source(source, output_file, func_globals=()):
    "Render the SSA graph given python source code"
    mod = ast.parse(source)
    func_ast = mod.body[0]

    env = environment.NumbaEnvironment.get_environment()
    func_env, _ = pipeline.run_pipeline2(env, None, func_ast, void(),
                                         pipeline_name='normalize',
                                         function_globals=dict(func_globals))

    env.translation.push_env(func_env)
    try:
        symtab, cfflow = entrypoints.build_ssa(env, func_env.ast)
    finally:
        env.translation.pop()

    render_ssa(cfflow, symtab, output_file)

# ______________________________________________________________________
# Test

if __name__ == '__main__':
    source = textwrap.dedent("""
        def func():
            x = 0
            for i in range(10):
                if i < 5:
                    x = i
                x += i

            y = x
            x = i
    """)

    render_ssa_from_source(source, os.path.expanduser("~/ssa.dot"))