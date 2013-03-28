# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ast
from itertools import imap, chain

from numba import nodes
from numba import symtab
from numba import visitors
from numba.nodes import cfnodes
from numba.traits import traits, Delegate
from numba.control_flow.cfstats import *
from numba.control_flow.debug import logger, debug

#------------------------------------------------------------------------
# Single Static Assignment (CFA stage)
#------------------------------------------------------------------------

@traits
class SSAifier(object):
    """
    Put the program in SSA form (single static assignment), given
    a control flow graph with abstract control flow statements.

    Computes the dominators for all basic blocks, computes the dominance
    frontier, renames the variables, creates the phis in the basic blocks,
    does reaching analysis (which is trivial at this point), and computes
    the def/use and use/def chains.

        def/use chain:
            cf_references

        use/def chain:
            In SSA form, only one definition can reach each use.
            The definition of a use is set by its 'variable' attribute.
    """

    blocks = Delegate('flow')

    def __init__(self, flow):
        self.flow = flow

    def compute_dominators(self):
        """
        Compute the dominators for the CFG, i.e. for each basic block the
        set of basic blocks that dominate that block. This mean from the
        entry block to that block must go through the blocks in the dominator
        set.

        dominators(x) = {x} ∪ (∩ dominators(y) for y ∈ preds(x))
        """
        blocks = set(self.blocks)
        for block in self.blocks:
            block.dominators = blocks

        changed = True
        while changed:
            changed = False
            for block in self.blocks:
                parent_dominators = [parent.dominators for parent in block.parents]
                new_doms = set.intersection(block.dominators, *parent_dominators)
                new_doms.add(block)

                if new_doms != block.dominators:
                    block.dominators = new_doms
                    changed = True

    def immediate_dominator(self, x):
        """
        The dominator of x that is dominated by all other dominators of x.
        This is the block that has the largest dominator set.
        """
        candidates = x.dominators - set([x])
        if not candidates:
            return None

        result = max(candidates, key=lambda b: len(b.dominators))
        ndoms = len(result.dominators)
        assert len([b for b in candidates if len(b.dominators) == ndoms]) == 1
        return result

    def compute_dominance_frontier(self):
        """
        Compute the dominance frontier for all blocks. This indicates for
        each block where dominance stops in the CFG. We use this as the place
        to insert Φ functions, since at the dominance frontier there are
        multiple control flow paths to the block, which means multiple
        variable definitions can reach there.
        """
        if debug:
            print("Dominator sets:")
            for block in self.blocks:
                print((block.id, sorted(block.dominators, key=lambda b: b.id)))

        blocks = []
        for block in self.blocks:
            if block.parents:
                block.idom = self.immediate_dominator(block)
                block.visited = False
                blocks.append(block)

        self.blocks = blocks

        def visit(block, result):
            block.visited = True
            for child in block.children:
                if not child.visited:
                    visit(child, result)
            result.append(block)

        #postorder = []
        #visit(self.blocks[0], postorder)
        postorder = self.blocks[::-1]

        # Compute dominance frontier
        for x in postorder:
            for y in x.children:
                if y.idom is not x:
                    # We are not an immediate dominator of our successor, add
                    # to frontier
                    x.dominance_frontier.add(y)

            for z in self.blocks:
                if z.idom is x:
                    for y in z.dominance_frontier:
                        if y.idom is not x:
                            x.dominance_frontier.add(y)

    def update_for_ssa(self, ast, symbol_table):
        """
        1) Compute phi nodes

            for each variable v
                1) insert empty phi nodes in dominance frontier of each block
                   that defines v
                2) this phi defines a new assignment in each block in which
                   it is inserted, so propagate (recursively)

        2) Reaching definitions

            Set block-local symbol table for each block.
            This is a rudimentary form of reaching definitions, but we can
            do it in a single pass because all assignments are known (since
            we inserted the phi functions, which also count as assignments).
            This means the output set is known up front for each block
            and never changes. After setting all output sets, we can compute
            the input sets in a single pass:

                1) compute output sets for each block
                2) compute input sets for each block

        3) Update phis with incoming variables. The incoming variables are
           last assignments of the predecessor blocks in the CFG.
        """
        # Print dominance frontier
        if debug:
            print("Dominance frontier:")
            for block in self.blocks:
                print(('DF(%d) = %s' % (block.id, block.dominance_frontier)))

        # Clear cf_references, we will only keep SSA references for each def
        for block in self.blocks:
            for stat in block.stats:
                if isinstance(stat, NameAssignment):
                    stat.entry.cf_references = []

        argnames = [name.id for name in ast.args.args]

        #
        ### 1) Insert phi nodes in the right places
        #
        for name, variable in symbol_table.iteritems():
            if not variable.renameable:
                continue

            defining = []
            for b in self.blocks:
                if variable in b.gen:
                    defining.append(b)

            for defining_block in defining:
                for f in defining_block.dominance_frontier:
                    phi = f.phis.get(variable, None)
                    if phi is None:
                        phi = PhiNode(f, variable)
                        f.phis[variable] = phi
                        defining.append(f)

        #
        ### 2) Reaching definitions and variable renaming
        #

        # Set originating block for each variable (as if each variable were
        # initialized at the start of the function) and start renaming of
        # variables
        symbol_table.counters = dict.fromkeys(symbol_table, -1) # var_name -> counter
        self.blocks[0].symtab = symbol_table
        for var_name, var in symbol_table.items():
            if var.renameable:
                new_var = symbol_table.rename(var, self.blocks[0])
                new_var.uninitialized = var.name not in argnames

        self.rename_assignments(self.blocks[0])

        for block in self.blocks[1:]:
            block.symtab = symtab.Symtab(parent=block.idom.symtab)
            for var, phi_node in block.phis.iteritems():
                phi_node.variable = block.symtab.rename(var, block)
                phi_node.variable.name_assignment = phi_node
                phi_node.variable.is_phi = True

            self.rename_assignments(block)

        #
        ### 3) Update the phis with all incoming entries
        #
        for block in self.blocks:
            # Insert phis in AST
            block.phi_nodes = block.phis.values()
            for variable, phi in block.phis.iteritems():
                for parent in block.parents:
                    incoming_var = parent.symtab.lookup_most_recent(variable.name)
                    phi.incoming.add(incoming_var)

                    phi.variable.uninitialized |= incoming_var.uninitialized

                    # Update def-use chain
                    incoming_var.cf_references.append(phi)

        #
        ### 4) Remove any unnecessary phis
        #
        kill_unused_phis(self.flow)

    def rename_assignments(self, block):
        lastvars = dict(block.symtab)
        for stat in block.stats:
            if (isinstance(stat, NameAssignment) and
                    stat.assignment_node and
                    stat.entry.renameable):
                # print "setting", stat.lhs, hex(id(stat.lhs))
                stat.lhs.variable = block.symtab.rename(stat.entry, block)
                stat.lhs.variable.name_assignment = stat
            elif isinstance(stat, NameReference) and stat.entry.renameable:
                current_var = block.symtab.lookup_most_recent(stat.entry.name)
                stat.node.variable = current_var
                current_var.cf_references.append(stat.node)

#------------------------------------------------------------------------
# Kill unused Phis
#------------------------------------------------------------------------

def kill_phi(block, phi):
    logger.debug("Killing phi: %s", phi)

    block.symtab.pop(phi.variable.renamed_name)

    for incoming_var in phi.incoming:
        # A single definition can reach a block multiple times,
        # remove all references
        refs = [ref for ref in incoming_var.cf_references
                        if ref.variable is not phi.variable]
        incoming_var.cf_references = refs

def kill_unused_phis(cfg):
    """
    Kill phis right after computing the SSA graph.

    Kills phis which are not referenced. We need to do this bottom-up,
    i.e. in reverse topological dominator-tree order, since in SSA
    a definition always lexically precedes a reference.

    This is important, since it kills any unnecessary promotions (e.g.
    ones to object, which LLVM wouldn't be able to optimize out).

    TODO: Kill phi cycles, or do reachability analysis before inserting phis.
    """
    changed = False

    for block in cfg.blocks[::-1]:
        phi_nodes = []

        for i, phi in enumerate(block.phi_nodes):
            if phi.variable.cf_references:
                # Used phi
                # logger.info("Used phi %s, %s" % (phi, phi.variable.cf_references))
                phi_nodes.append(phi)
            else:
                # Unused phi
                changed = True

                kill_phi(block, phi)

        block.phi_nodes = phi_nodes

    return changed

#------------------------------------------------------------------------
# Merge Phis into AST (CFA stage)
#------------------------------------------------------------------------

def phi_nodes(basic_block):
    return basic_block.phis.values()

def merge(basic_block, node):
    """
    Merge the phis of the given basic block in the given node.
    Returns a new node.
    """
    return ast.Suite(phi_nodes(basic_block) + [node])

def merge_inplace(basic_block, body_list):
    """
    In-place merge of phi nodes from a basic block into a list of AST
    statements.
    """
    nodes = body_list[:]
    body_list.clear()
    body_list.extend(phi_nodes(basic_block))
    body_list.extend(nodes)


class PhiInjector(visitors.NumbaTransformer):
    """
    Merge PHIs into AST. This happens after the CFG was build and the
    phis computed.
    """

    def visit_FunctionDef(self, node):
        self.visitchildren(node)
        merge_inplace(node.body_block, node.body)
        node.body.extend(phi_nodes(self.env.crnt.flow.exit_point))
        return node

    def handle_if_or_while(self, node, body_block):
        self.visitchildren(node)

        # condition
        node.test = nodes.ExpressionNode(phi_nodes(node.cond_block), node.test)

        # body
        merge_inplace(body_block, node.body)

        # exit block
        postceding = phi_nodes(node.exit_block)

        return ast.Suite([node] + postceding)

    def visit_If(self, node):
        return self.handle_if_or_while(node, node.if_block)

    def visit_While(self, node):
        return self.handle_if_or_while(node, node.while_block)

    def visit_For(self, node):
        self.visitchildren(node)

        # condition
        preceding = phi_nodes(node.cond_block)

        # body
        merge_inplace(node.for_block, node.body)

        # incr
        incr = phi_nodes(node.incr_block)

        # exit block
        postceding = phi_nodes(node.exit_block)

        for_node = cfnodes.For(node.target, node.iter, node.body,
                               incr, node.orelse)
        return ast.Suite(preceding + [for_node] + postceding)



#------------------------------------------------------------------------
# Iterate over all phi nodes or variables
#------------------------------------------------------------------------

def iter_phis(flow):
    "Iterate over all phi nodes"
    return chain(*[block.phi_nodes for block in flow.blocks])

def iter_phi_vars(flow):
    "Iterate over all phi nodes"
    for phi_node in iter_phis(flow):
        yield phi_node.variable

#------------------------------------------------------------------------
# Specialization code for SSA
#------------------------------------------------------------------------

# TODO: Do this before spitting out typed IR

def specialize_ssa(funcdef):
    """
    Handle phi nodes:

        1) Handle incoming variables which are not initialized. Set
           incoming_variable.uninitialized_value to a constant 'bad'
           value (e.g. 0xbad for integers, NaN for floats, NULL for
           objects)

        2) Handle incoming variables which need promotions. An incoming
           variable needs a promotion if it has a different type than
           the the phi. The promotion happens in each ancestor block that
           defines the variable which reaches us.

           Promotions are set separately in the symbol table, since the
           ancestor may not be our immediate parent, we cannot introduce
           a rename and look up the latest version since there may be
           multiple different promotions. So during codegen, we first
           check whether incoming_type == phi_type, and otherwise we
           look up the promotion in the parent block or an ancestor.
    """
    for phi_node in iter_phis(funcdef.flow):
        specialize_phi(phi_node)

def specialize_phi(node):
    for parent_block, incoming_var in node.find_incoming():
        if incoming_var.type.is_uninitialized:
            incoming_type = incoming_var.type.base_type or node.type
            bad = nodes.badval(incoming_type)
            incoming_var.type.base_type = incoming_type
            incoming_var.uninitialized_value = bad
            # print incoming_var

        elif not incoming_var.type == node.type:
            # Create promotions for variables with phi nodes in successor
            # blocks.
            incoming_symtab = incoming_var.block.symtab
            if (incoming_var, node.type) not in node.block.promotions:
                # Make sure we only coerce once for each destination type and
                # each variable
                incoming_var.block.promotions.add((incoming_var, node.type))

                # Create promotion node
                name_node = nodes.Name(id=incoming_var.renamed_name,
                                       ctx=ast.Load())
                name_node.variable = incoming_var
                name_node.type = incoming_var.type
                coercion = name_node.coerce(node.type)
                promotion = nodes.PromotionNode(node=coercion)

                # Add promotion node to block body
                incoming_var.block.body.append(promotion)
                promotion.variable.block = incoming_var.block

                # Update symtab
                incoming_symtab.promotions[incoming_var.name,
                                           node.type] = promotion
            else:
                promotion = incoming_symtab.lookup_promotion(
                    incoming_var.name, node.type)

    return node

#------------------------------------------------------------------------
# Handle phis during code generation
#------------------------------------------------------------------------

# TODO: should will be explicit in the IR, remove

def process_incoming(phi_node):
    """
    Add all incoming phis to the phi instruction.

    Handle promotions by using the promoted value from the incoming block.
    E.g.

        bb0: if C:
        bb1:     x = 2
             else:
        bb2:     x = 2.0

        bb3: x = phi(x_bb1, x_bb2)

    has a promotion for 'x' in basic block 1 (from int to float).
    """
    var = phi_node.variable
    phi = var.lvalue

    for parent_block, incoming_var in phi_node.find_incoming():
        if incoming_var.type.is_uninitialized:
            pass
        elif not incoming_var.type == phi_node.type:
            promotion = parent_block.symtab.lookup_promotion(var.name,
                                                             phi_node.type)
            incoming_var = promotion.variable

        assert incoming_var.lvalue, incoming_var
        assert parent_block.exit_block, parent_block

        phi.add_incoming(incoming_var.lvalue,
                         parent_block.exit_block)

        if phi_node.type.is_array:
            nodes.update_preloaded_phi(phi_node.variable,
                                       incoming_var,
                                       parent_block.exit_block)


def handle_phis(flow):
    """
    Update all our phi nodes after translation is done and all Variables
    have their llvm values set.
    """
    if flow is None:
        return

    for phi_node in iter_phis(flow):
        process_incoming(phi_node)
