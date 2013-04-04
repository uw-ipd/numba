# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

import ast

from collections import defaultdict
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
            # Insert phis in CFG
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

def inject_phis(env, cfg, ast):
    "Inject phis as PhiNode nodes into the AST"
    injector = PhiInjector(env.context, None, ast, env, cfg)
    return injector.visit(ast)

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
    body_list[:] = phi_nodes(basic_block) + nodes

class PhiInjector(visitors.NumbaTransformer):
    """
    Merge PHIs into AST. This happens after the CFG was build and the
    phis computed.
    """

    def __init__(self, context, func, ast, env, cfg, **kwargs):
        super(PhiInjector, self).__init__(context, func, ast, env, **kwargs)
        self.cfg = cfg

    def visit_FunctionDef(self, node):
        self.visitchildren(node)
        merge_inplace(node.body_block, node.body)
        node.body.extend(phi_nodes(self.cfg.exit_point))
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

        preceding = phi_nodes(node.cond_block)
        merge_inplace(node.for_block, node.body)
        postceding = phi_nodes(node.exit_block)

        for_node = cfnodes.For(node.target, node.iter, node.body, node.orelse)
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

def specialize_ssa(env, ast):
    """
    Handle phi nodes:

        1) Handle incoming variables which are not initialized. Create
           constant 'bad' values (e.g. 0xbad for integers, NaN for floats,
           NULL for objects)

        2) Handle incoming variables which need promotions. An incoming
           variable needs a promotion if it has a different type than
           the the phi. The promotion happens in each ancestor block that
           defines the variable which reaches us.
    """
    # variable_def -> BadValue
    badvals = {}

    # variable_def -> dst_type -> [PromotionNode]
    # Remember that multiple phis may reference a single other phi
    promotions = defaultdict(dict)

    for phi_node in iter_phis(env.crnt.cfg):
        initialize_uninitialized(phi_node, badvals)
        promote_incoming(phi_node, promotions)

    ast = merge_badvals(env, ast, badvals)
    ast = merge_promotions(env, ast, promotions)

    return ast

def initialize_uninitialized(phi_node, badvals):
    """
    Initialize potentially uninitialized variables. For instance:

        def func(A):
            if ...:
                x = 0

            if ...:
                use(x)

    'x' may be uninitialized when it is used. So we find the implicit top
    definition of 'x' ('x_0'), and assign a bad value.

    We can find variables that need such initialization in two ways:

        1) For each NameAssignment in the CFG find the variables of type
           Uninitialized and live cf_references.

        2) For each phi in the CFG see whether there is an incoming variable
           with type Uninitialized. This finds all Uninitialized variables
           since if the variable was /not/ an incoming variable of some phi,
           then the variable was certainly referenced before assignment,
           and we would have issued an error.
    """
    for parent_block, incoming_var in phi_node.find_incoming():
        if incoming_var.type.is_uninitialized:
            incoming_type = incoming_var.type.base_type or phi_node.type
            bad = nodes.badval(incoming_type)
            incoming_var.type.base_type = incoming_type

            badvals[incoming_var] = bad
            # incoming_var.uninitialized_value = bad

def merge_badvals(env, func_ast, badvals):
    assmnts = []
    for var in badvals:
        name = nodes.Name(id=var.renamed_name, ctx=ast.Store())
        name.variable = var
        assmnts.append(ast.Assign(targets=[name], value=badvals[var]))

    func_ast.body = assmnts + func_ast.body
    return func_ast

def promote_incoming(phi_node, promotions):
    """
    Promote incoming definitions:

        x = 0
        if ...:
            x = 1.0
        # phi(x_1, x_2)
        print x

    We need to promote the definition 'x = 0' to double, i.e.:

        x = 0
          ->  [ x = 0; %0 = double(x) ]

        phi(x_1, x_2)
          -> phi(%0, x_2)
    """
    for parent_block, incoming_var in phi_node.find_incoming():
        is_uninitialized = incoming_var.type.is_uninitialized
        if not is_uninitialized and incoming_var.type != phi_node.type:
            # Create promotions for variables with phi nodes in successor
            # blocks.
            if phi_node.type not in promotions[incoming_var]:
                # Make sure we only coerce once for each destination type and
                # each variable

                # Create promotion phi_node
                name_node = nodes.Name(id=incoming_var.renamed_name,
                                       ctx=ast.Load())
                name_node.variable = incoming_var
                name_node.type = incoming_var.type

                coercion = name_node.coerce(phi_node.type)
                promotion = nodes.PromotionNode(node=coercion)

                promotions[incoming_var, phi_node.type] = promotion

def merge_promotions(env, ast, promotions):
    merger = PromotionMerger(env.context, env.crnt.func, ast, env, promotions)
    return merger.visit(ast)

class PromotionMerger(visitors.NumbaTransformer):
    """
    Merge in promotions in AST.
    Promotions are generated by promote_incoming().
    """

    # TODO: This is better done on lowered IR

    def __init__(self, context, func, ast, env, promotions, **kwargs):
        super(PromotionMerger, self).__init__(context, func, ast, env, **kwargs)
        self.promotions = promotions

    def visit_Name(self, node):
        if (isinstance(node.ctx, ast.Store) and
                self.promotions[node.variable]):
            promotion_nodes = self.promotions[node.variable]
            node = nodes.ExpressionNode([node] + promotion_nodes, node)

        return node

    def visit_PhiNode(self, node):
        for incoming_def in node.incoming:
            if node.type in self.promotions[incoming_def]:
                promotion = self.promotions[incoming_def][node.type]
                node.incoming.remove(incoming_def)
                node.incoming.add(promotion.variable)

        return node

#------------------------------------------------------------------------
# Handle phis during code generation
#------------------------------------------------------------------------

def propagate_defs(block):
    parent_defs = {}
    for parent in block.parents:
        parent_defs.update(parent.live_defs)
    block.live_defs = dict(parent_defs, **block.live_defs)

def propagate_promotions(block):
    for parent in block.parents:
        block.live_promotions.update(parent.live_promotions)

def find_parent(block, phi_def):
    for parent in block.parents:
        incoming = parent.get(phi_def.name, None)
        if incoming is phi_def or phi_def in parent.live_promotions:
            return parent

    assert False

def preload_array_attrs(phi_def, incoming, parent_block):
    if phi_def.type.is_array:
        nodes.update_preloaded_phi(phi_def,
                                   incoming,
                                   parent_block.llvm_block)

def update_ssa_graph(flow):
    """
    Add all incoming phis to the phi instruction.
    """
    # Forward propagate promotions and definitions
    for block in flow.blocks:
        propagate_defs(block)
        propagate_promotions(block)

    # Update LLVM SSA graph
    for block in flow.blocks:
        for phi_def in block.phi_defs:
            phi = phi_def.lvalue
            for incoming in phi_def.incoming:
                parent_block = find_parent(block, incoming)
                phi.add_incoming(incoming.lvalue, parent_block.llvm_block)
                preload_array_attrs(phi_def, incoming, parent_block)
