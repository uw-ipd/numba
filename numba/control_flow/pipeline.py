# -*- coding: utf-8 -*-

"""
Control flow for the AST backend.

Adapted from Cython/Compiler/FlowControl.py
"""
from __future__ import print_function, division, absolute_import

import re
import ast
import copy
from functools import reduce

from numba import error, visitors, symtab, nodes, reporting

from numba import *
from numba.control_flow import ssa
from numba.control_flow import flow
from numba.control_flow import control_flow
from numba.control_flow import graphviz
from numba.control_flow import reaching
from numba.control_flow import cfwarnings
from numba.control_flow.cfstats import *
from numba.control_flow.debug import *


def control_flow_pipeline(env, ast):
    # Cleanup graph
    # self.flow.normalize()

    messages = env.crnt.error_env.collection
    self.warner = cfwarnings.CFWarner(messages, self.current_directives)

    if env:
        if hasattr(env, 'translation'):
            env.translation.crnt.cfg_transform = self

    self.graphviz = self.current_directives['control_flow.dot_output']
    if self.graphviz:
        self.gv_ctx = graphviz.GVContext()
        self.source_descr = reporting.SourceDescr(func, ast)


    reaching.check_definitions(self.env, self.flow, self.warner)

    # self.render_gv(node)

    ssa_maker = ssa.SSAifier(self.flow)
    ssa_maker.compute_dominators()
    ssa_maker.compute_dominance_frontier()
    ssa_maker.update_for_ssa(self.ast, self.symtab)