# -*- coding: utf-8 -*-

"""
Control flow for the AST backend.

Adapted from Cython/Compiler/FlowControl.py
"""
from __future__ import print_function, division, absolute_import

import ast

from numba import visitors
from numba import symtab
from numba import nodes
from numba.control_flow.cfstats import *

class ControlFlowAnalysis(visitors.NumbaTransformer):
    """
    Control flow analysis pass that builds the CFG and injects the blocks
    into the AST (but not all blocks are injected).

    The CFG must be build in topological dominator tree order, e.g. the
    'if' condition block must precede the clauses and the clauses must
    precede the exit.
    """

    function_level = 0

    def __init__(self, env, func, ast, cfflow):
        super(ControlFlowAnalysis, self).__init__(env.context, func, ast,
                                                  env)
        self.flow = cfflow
        self.symtab = self.initialize_symtab(allow_rebind_args=True)

    # ______________________________________________________________________

    # TODO: Make this a separate pass

    def initialize_symtab(self, allow_rebind_args):
        """
        Populate the symbol table with variables and set their renaming status.

        Variables appearing in locals, or arguments typed through the 'jit'
        decorator are not renameable.
        """
        symbols = symtab.Symtab(self.symtab)
        for var_name in self.local_names:
            variable = symtab.Variable(None, name=var_name, is_local=True)

            # Set cellvar status. Free variables are not assignments, and
            # are caught in the type inferencer
            variable.is_cellvar = var_name in self.cellvars
            # variable.is_freevar = var_name in self.freevars

            variable.renameable = (
                var_name not in self.locals and not
                (variable.is_cellvar or variable.is_freevar) and
                (var_name not in self.argnames or allow_rebind_args))

            symbols[var_name] = variable

        return symbols

    # ______________________________________________________________________

    def visit(self, node):
        if hasattr(node, 'lineno'):
            self.mark_position(node)

        if not self.flow.block:
            # Unreachable code
            # NOTE: removing this here means there is no validation of the
            # unreachable code!
            self.warner.warn_unreachable(node)
            return None

        return super(ControlFlowAnalysis, self).visit(node)

    def visit_FunctionDef(self, node):
        #for arg in node.args:
        #    if arg.default:
        #        self.visitchildren(arg)
        if self.function_level:
            return node

        self.function_level += 1

        self.visitlist(node.decorator_list)

        # Collect all entries
        for var_name, var in self.symtab.iteritems():
            if var_name not in self.locals:
                self.flow.entries.add(var)

        self.flow.nextblock(node, 'entry')
        self.mark_position(node)

        # Function body block
        node.body_block = self.flow.nextblock(node, 'function_body')
        for arg in node.args.args:
            if hasattr(arg, 'id') and hasattr(arg, 'ctx'):
                self.visit_Name(arg)
            else:
                self.visit_arg(arg, node.lineno, 0)

        self.visitlist(node.body)
        self.function_level -= 1

        # Exit point
        if self.flow.block:
            self.flow.block.add_child(self.flow.exit_point)

        self.flow.add_floating(self.flow.exit_point)

        return node

    def mark_assignment(self, lhs, rhs=None, assignment=None, warn_unused=True):
        assert self.flow.block

        if self.flow.exceptions:
            exc_descr = self.flow.exceptions[-1]
            self.flow.block.add_child(exc_descr.entry_point)
            self.flow.nextblock()

        if not rhs:
            rhs = None

        lhs = self.visit(lhs)
        name_assignment = None
        if isinstance(lhs, ast.Name):
            name_assignment = self.flow.mark_assignment(
                    lhs, rhs, self.symtab[lhs.name], assignment,
                    warn_unused=warn_unused)

        # TODO: Generate fake RHS for for iteration target variable
        elif (isinstance(lhs, ast.Attribute) and self.flow.block and
                  assignment is not None):
            self.flow.block.stats.append(AttributeAssignment(assignment))

        if self.flow.exceptions:
            exc_descr = self.flow.exceptions[-1]
            self.flow.block.add_child(exc_descr.entry_point)
            self.flow.nextblock()

        return lhs, name_assignment

    def mark_position(self, node):
        """Mark position if DOT output is enabled."""
        if self.current_directives['control_flow.dot_output']:
            self.flow.mark_position(node)

    def visit_Assign(self, node):
        node.value = self.visit(node.value)
        if len(node.targets) == 1 and isinstance(node.targets[0],
                                                 (ast.Tuple, ast.List)):
            node.targets = node.targets[0].elts

        for i, target in enumerate(node.targets):
            # target = self.visit(target)

            maybe_unused_node = isinstance(target, nodes.MaybeUnusedNode)
            if maybe_unused_node:
                target = target.name_node

            lhs, name_assignment = self.mark_assignment(target, node.value,
                                                        assignment=node,
                                                        warn_unused=not maybe_unused_node)
            node.targets[i] = lhs

        return node

    def visit_arg(self, old_node, lineno, col_offset):
        node = nodes.Name(old_node.arg, ast.Param())
        node.lineno = lineno
        node.col_offset = col_offset
        return self._visit_Name(node)

    def visit_Name(self, old_node):
        node = nodes.Name(old_node.id, old_node.ctx)
        ast.copy_location(node, old_node)
        return self._visit_Name(node)

    def _visit_Name(self, node):
        # Set some defaults
        node.cf_maybe_null = True
        node.cf_is_null = False
        node.allow_null = False

        node.name = node.id
        if isinstance(node.ctx, ast.Param):
            var = self.symtab[node.name]
            var.is_arg = True
            self.flow.mark_assignment(node, None, var, assignment=None)
        elif isinstance(node.ctx, ast.Load):
            var = self.symtab.lookup(node.name)
            if var:
                # Local variable
                self.flow.mark_reference(node, var)

        # Set position of assignment of this definition
        if isinstance(node.ctx, (ast.Param, ast.Store)):
            var = self.symtab[node.name]
            if var.lineno == -1:
                var.lineno = getattr(node, "lineno", 0)
                var.col_offset = getattr(node, "col_offset", 0)

        return node

    def visit_MaybeUnusedNode(self, node):
        self.symtab[node.name_node.id].warn_unused = False
        return self.visit(node.name_node)

    def visit_Suite(self, node):
        if self.flow.block:
            for i, stat in enumerate(node.body):
                node.body[i] = self.visit(stat)
                if not self.flow.block:
                    stat.is_terminator = True
                    break

        return node

    def visit_ImportFrom(self, node):
        for name, target in node.names:
            if name != "*":
                self.mark_assignment(target, assignment=node)

        self.visitchildren(node)
        return node

    def visit_If(self, node):
        with self.flow.float(node, 'exit_if') as node.exit_block:
            # Condition
            node.cond_block = self.flow.nextblock(node.test, 'if_cond')
            node.test = self.visit(node.test)

            self.handle_body(node, node.exit_block)
            self.handle_else_clause(node, node.cond_block, node.exit_block)

        return node

    def handle_body(self, node, exit_block):
        # If Body, child of condition block
        node.if_block = self.flow.nextblock(node.body[0], 'body')
        self.visitlist(node.body)

        if self.flow.block:
            self.flow.block.add_child(exit_block)

    def handle_else_clause(self, node, cond_block, exit_block):
        if node.orelse:
            # Else clause, child of condition block
            node.else_block = self.flow.nextblock(node.orelse[0],
                                                  'else_body', cond_block)
            self.visitlist(node.orelse)

            if self.flow.block:
                self.flow.block.add_child(exit_block)
        else:
            # No else clause, the exit block is a child of the condition
            cond_block.add_child(exit_block)
            node.else_block = None

    def finalize_loop(self, node, cond_block, exit_block):
        if self.flow.block:
            # Add back-edge
            self.flow.block.add_child(cond_block)

        self.handle_else_clause(node, cond_block, exit_block)

    def visit_While(self, node):
        with self.flow.float(node, 'exit_while') as node.exit_block:
            # Condition block
            node.cond_block = self.flow.nextblock(node.test, 'while_condition')
            self.flow.loops.append(LoopDescr(node.exit_block, node.cond_block))
            node.test = self.visit(node.test)

            # Body
            node.while_block = self.flow.nextblock(node.body[0], "while_body")
            self.visitlist(node.body)
            self.flow.loops.pop()

            self.finalize_loop(node, node.cond_block, node.exit_block)

        return node

    def visit_For(self, node):
        # Evaluate iterator in previous block
        node.iter = self.visit(node.iter)

        # Start condition block
        node.cond_block = self.flow.nextblock(node.iter, 'for_condition')

        # Increment temporary variable, continue should branch here
        node.incr_block = self.flow.newblock(label="for_increment", pos=node)

        with self.flow.float(node, 'exit_for') as node.exit_block:
            # Body
            node.for_block = self.flow.nextblock(node.body[0], 'for_body')
            self.flow.loops.append(LoopDescr(node.exit_block, node.incr_block))

            # Assign to target variable in body
            node.target, name_assignment = self.mark_assignment(
                node.target, assignment=None, warn_unused=False)

            self.visitlist(node.body)
            self.flow.loops.pop()

            if self.flow.block:
                self.flow.block.add_child(node.incr_block)

            # Ensure topological dominator order
            self.flow.blocks.pop(node.incr_block.id)
            self.flow.blocks.append(node.incr_block)

            self.flow.block = node.incr_block

            self.finalize_loop(node, node.cond_block, node.exit_block)

            if name_assignment:
                name_assignment.assignment_node = node

        return node

    def visit_With(self, node):
        node.context_expr = self.visit(node.context_expr)
        if node.optional_vars:
            # TODO: Mark these as assignments!
            node.optional_vars = self.visit(node.optional_vars)

        # TODO: Build CFG for with blocks

        self.visitlist(node.body)
        return node

    def visit_Raise(self, node):
        self.visitchildren(node)
        if self.flow.exceptions:
            self.flow.block.add_child(self.flow.exceptions[-1].entry_point)

        self.flow.block = None
        return node

    def visit_Return(self, node):
        self.visitchildren(node)

        for exception in self.flow.exceptions[::-1]:
            if exception.finally_enter:
                self.flow.block.add_child(exception.finally_enter)
                if exception.finally_exit:
                    exception.finally_exit.add_child(self.flow.exit_point)
                break
        else:
            if self.flow.block:
                self.flow.block.add_child(self.flow.exit_point)

        self.flow.block = None
        return node

    def visit_Break(self, node):
        if not self.flow.loops:
            return node

        loop = self.flow.loops[-1]
        for exception in loop.exceptions[::-1]:
            if exception.finally_enter:
                self.flow.block.add_child(exception.finally_enter)
                if exception.finally_exit:
                    exception.finally_exit.add_child(loop.next_block)
                break
        else:
            self.flow.block.add_child(loop.next_block)

        self.flow.block = None
        return node

    def visit_Continue(self, node):
        if not self.flow.loops:
            return node

        loop = self.flow.loops[-1]
        for exception in loop.exceptions[::-1]:
            if exception.finally_enter:
                self.flow.block.add_child(exception.finally_enter)
                if exception.finally_exit:
                    exception.finally_exit.add_child(loop.loop_block)
                break
        else:
            self.flow.block.add_child(loop.loop_block)

        self.flow.block = None
        return node

    def visit_Print(self, node):
        self.generic_visit(node)
        return node
