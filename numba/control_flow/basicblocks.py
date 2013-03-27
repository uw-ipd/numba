# -*- coding: utf-8 -*-

"""
Control flow basic blocks.
"""

from __future__ import print_function, division, absolute_import

#----------------------------------------------------------------------------
# CFG basic blocks
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

        self.children = set()
        self.parents = set()
        self.positions = set()

    def detach(self):
        """Detach block from parents and children."""
        for child in self.children:
            child.parents.remove(self)
        for parent in self.parents:
            parent.children.remove(self)
        self.parents.clear()
        self.children.clear()

    def add_child(self, block):
        self.children.add(block)
        block.parents.add(self)

    def reparent(self, new_block):
        """
        Re-parent all children to the new block
        """
        for child in self.children:
            child.parents.remove(self)
            new_block.add_child(child)

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
        return 'Block(%d)' % self.id
