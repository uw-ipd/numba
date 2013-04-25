# -*- coding: utf-8 -*-

"""
Control flow basic blocks.
"""

from __future__ import print_function, division, absolute_import

from numba.oset import OrderedSet

#----------------------------------------------------------------------------
# CFG basic block interface
#----------------------------------------------------------------------------

class BasicBlock(object):
    """
    Control flow graph node.

        children:  set of children nodes
        parents:   set of parent nodes

    This is the interface that specifies the core facilities for basic
    blocks.

    Subclasses can for instance add:

        * abstract control flow statements,
        * localized symbol tables
        * code generation in terms of labels and branches
    """

    def __init__(self, id, label, pos):
        self.id = id
        self.label = label
        self.pos = pos

        self.positions = set()
        self.children = OrderedSet()
        self.parents = OrderedSet()

    def detach(self):
        """Detach block from parents and children."""
        self.detach_parents()
        self.detach_children()

    def detach_parents(self):
        for parent in self.parents:
            parent.children.remove(self)
        self.parents.clear()

    def detach_children(self):
        for child in self.children:
            child.parents.remove(self)
        self.children.clear()

    def add_parents(self, *parents):
        for parent in parents:
            self.parents.add(parent)
            parent.children.add(self)

    def add_children(self, *children):
        for child in children:
            self.children.add(child)
            child.parents.add(self)

    def reparent(self, new_block):
        """
        Re-parent all children to the new block
        """
        for child in self.children:
            child.parents.remove(self)
            new_block.add_children(child)

    def delete(self, flow):
        """
        Delete a block from the cfg.
        """
        for parent in self.parents:
            parent.children.remove(self)
        for child in self.children:
            child.parents.remove(self)

        flow.blocks.remove(self)

    def __repr__(self):
        return 'Block(%d, %s)' % (self.id, self.label)
