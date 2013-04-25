# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

from numba.traits import traits, Delegate
from numba.control_flow.cfstats import (
    NameReference, NameAssignment, Uninitialized, PhiNode)


def allow_null(node):
    return False


def check_definitions(env, flow, warner):
    """
    Compute reaching definitions and check validity.

    Adapted from Cython/Compiler/FlowControl
    """

    # ______________________________________________________________________
    # Compute reaching defs
    reacher = ReachingDefs(flow)

    reacher.initialize()
    reacher.reaching_definitions()

    # Track down state
    assignments = set()
    # Node to entry map
    references = {}
    assmt_nodes = set()

    for block in flow.blocks:
        i_state = block.i_input
        for stat in block.stats:
            if not isinstance(stat, (NameAssignment, NameReference)):
                continue

            i_assmts = reacher.assmts[stat.entry]
            state = reacher.map_one(i_state, stat.entry)
            if isinstance(stat, NameAssignment):
                stat.lhs.cf_state.update(state)
                assmt_nodes.add(stat.lhs)
                i_state = i_state & ~i_assmts.mask
                if stat.is_deletion:
                    i_state |= i_assmts.bit
                else:
                    i_state |= stat.bit
                assignments.add(stat)
                # if stat.rhs is not fake_rhs_expr:
                stat.entry.cf_assignments.append(stat)
            elif isinstance(stat, NameReference):
                references[stat.node] = stat.entry
                stat.entry.cf_references.append(stat)
                stat.node.cf_state.update(state)
                if not allow_null(stat.node):
                    i_state &= ~i_assmts.bit
                state.discard(Uninitialized)
                for assmt in state:
                    assmt.refs.add(stat)

    # assignment hints
    for node in assmt_nodes:
        maybe_null = Uninitialized in node.cf_state
        node.cf_maybe_null = maybe_null
        node.cf_is_null = maybe_null and len(node.cf_state) == 1

    # ______________________________________________________________________
    # Issue warnings

    warner.check_uninitialized(references)
    warner.warn_unused_result(assignments)
    warner.warn_unused_entries(flow)

    if warner.have_errors:
        warner.messages.report(post_mortem=False)

    for node in assmt_nodes:
        node.cf_state = None #ControlFlowState(node.cf_state)
    for node in references:
        node.cf_state = None #ControlFlowState(node.cf_state)


#------------------------------------------------------------------------
# Compute Reaching Definitions
#------------------------------------------------------------------------

class AssignmentList(object):
    def __init__(self):
        self.stats = []


@traits
class ReachingDefs(object):

    entry_point = Delegate('flow')
    blocks = Delegate('flow')
    entries = Delegate('flow')

    def __init__(self, flow):
        self.flow = flow
        self.assmts = None

    def normalize(self):
        """Delete unreachable and orphan blocks."""
        blocks = set(self.blocks)
        queue = set([self.entry_point])
        visited = set()
        while queue:
            root = queue.pop()
            visited.add(root)
            for child in root.children:
                if child not in visited:
                    queue.add(child)
        unreachable = blocks - visited
        for block in unreachable:
            block.detach()
        visited.remove(self.entry_point)
        for block in visited:
            if block.empty():
                for parent in block.parents: # Re-parent
                    for child in block.children:
                        parent.add_children(child)
                block.detach()
                unreachable.add(block)
        blocks -= unreachable
        self.blocks = [block for block in self.blocks if block in blocks]

    def initialize(self):
        """Set initial state, map assignments to bits."""
        self.assmts = {}

        offset = 0
        for entry in self.entries:
            assmts = AssignmentList()
            assmts.bit = 1 << offset
            assmts.mask = assmts.bit
            self.assmts[entry] = assmts
            offset += 1

        for block in self.blocks:
            block.stats = block.phis.values() + block.stats
            for stat in block.stats:
                if isinstance(stat, (PhiNode, NameAssignment)):
                    stat.bit = 1 << offset
                    assmts = self.assmts[stat.entry]
                    assmts.stats.append(stat)
                    assmts.mask |= stat.bit
                    offset += 1

        for block in self.blocks:
            for entry, stat in block.gen.items():
                assmts = self.assmts[entry]
                if stat is Uninitialized:
                    block.i_gen |= assmts.bit
                else:
                    block.i_gen |= stat.bit
                block.i_kill |= assmts.mask
            block.i_output = block.i_gen
            for entry in block.bound:
                block.i_kill |= self.assmts[entry].bit

        for assmts in self.assmts.itervalues():
            self.entry_point.i_gen |= assmts.bit
        self.entry_point.i_output = self.entry_point.i_gen

    def map_one(self, istate, entry):
        "Map the bitstate of a variable to the definitions it represents"
        ret = set()
        assmts = self.assmts[entry]
        if istate & assmts.bit:
            ret.add(Uninitialized)
        for assmt in assmts.stats:
            if istate & assmt.bit:
                ret.add(assmt)
        return ret

    def reaching_definitions(self):
        """Per-block reaching definitions analysis."""
        dirty = True
        while dirty:
            dirty = False
            for block in self.blocks:
                i_input = 0
                for parent in block.parents:
                    i_input |= parent.i_output
                i_output = (i_input & ~block.i_kill) | block.i_gen
                if i_output != block.i_output:
                    dirty = True
                block.i_input = i_input
                block.i_output = i_output
