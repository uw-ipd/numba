# -*- coding: utf-8 -*-

"""
Control flow for the AST backend.

Adapted from Cython/Compiler/FlowControl.py
"""

from __future__ import print_function, division, absolute_import

from numba.control_flow import ssa
from numba.control_flow import flow
from numba.control_flow import control_flow
from numba.control_flow import reaching
from numba.control_flow import cfwarnings

#---------------------------------------------------------------------------
# Control Flow Pipeline
#---------------------------------------------------------------------------

def build_cfg(ast, cfflow, env):
    cfg_builder = control_flow.ControlFlowAnalysis(
        env, env.crnt.func, ast, cfflow)

    ast = cfg_builder.visit(ast)
    env.crnt.ast = ast

    return ast, cfg_builder.symtab


def build_ssa(env, ast):
    # Cleanup graph
    # self.flow.normalize()

    # Handle warnings and errors
    messages = env.crnt.error_env.collection
    warner = cfwarnings.CFWarner(messages, env.crnt.cfdirectives)

    # Graphviz
    dotfile = env.crnt.cfdirectives['control_flow.dot_output']

    # Build CFG
    cfflow = flow.CFGFlow(env)
    ast, symtab = build_cfg(ast, cfflow, env)

    # Run reaching defs and issue warnings/errors
    reaching.check_definitions(env, cfflow, warner)

    # Build SSA graph
    ssa_maker = ssa.SSAifier(cfflow)
    ssa_maker.compute_dominators()
    ssa_maker.compute_dominance_frontier()
    ssa_maker.update_for_ssa(ast, symtab)

    return symtab, cfflow
