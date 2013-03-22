# -*- coding: utf-8 -*-

"""
Handle CFA warnings.
"""

from __future__ import print_function, division, absolute_import

from numba import error
from numba import traits
from numba.control_flow import reaching
from numba.control_flow.cfstats import Uninitialized

@traits.traits
class CFWarner(object):
    "Generate control flow related warnings."

    have_errors = traits.Delegate('messages')

    def __init__(self, message_collection, directives):
        self.messages = message_collection
        self.directives = directives

    def check_uninitialized(self, references):
        "Find uninitialized references and cf-hints"
        warn_maybe_uninitialized = self.directives['warn.maybe_uninitialized']

        for node, entry in references.iteritems():
            if Uninitialized in node.cf_state:
                node.cf_maybe_null = True
                from_closure = False # entry.from_closure
                if not from_closure and len(node.cf_state) == 1:
                    node.cf_is_null = True
                if reaching.allow_null(node) or from_closure:
                    pass # Can be uninitialized here
                elif node.cf_is_null:
                    is_object = True #entry.type.is_pyobject
                    is_unspecified = False #entry.type.is_unspecified
                    error_on_uninitialized = False #entry.error_on_uninitialized
                    if entry.renameable and (is_object or is_unspecified or
                                                 error_on_uninitialized):
                        self.messages.error(
                            node,
                            "local variable '%s' referenced before assignment"
                            % entry.name)
                    else:
                        self.messages.warning(
                            node,
                            "local variable '%s' referenced before assignment"
                            % entry.name)
                elif warn_maybe_uninitialized:
                    self.messages.warning(
                        node,
                        "local variable '%s' might be referenced before assignment"
                        % entry.name)
            else:
                node.cf_is_null = False
                node.cf_maybe_null = False

    def warn_unused_entries(self, flow):
        """
        Generate warnings for unused variables or arguments. This is issues when
        an argument or variable is unused entirely in the function.
        """
        warn_unused = self.directives['warn.unused']
        warn_unused_arg = self.directives['warn.unused_arg']

        for entry in flow.entries:
            if (not entry.cf_references and not entry.is_cellvar and
                    entry.renameable): # and not entry.is_pyclass_attr
                if entry.is_arg:
                    if warn_unused_arg:
                        self.messages.warning(
                            entry, "Unused argument '%s'" % entry.name)
                else:
                    if (warn_unused and entry.warn_unused and
                            flow.is_tracked(entry)):
                        if getattr(entry, 'lineno', 1) > 0:
                            self.messages.warning(
                                entry, "Unused variable '%s'" % entry.name)
                entry.cf_used = False

    def warn_unused_result(self, assignments):
        """
        Warn about unused variable definitions. This is issued for individual
        definitions, e.g.

            i = 0   # this definition generates a warning
            i = 1
            print i
        """
        warn_unused_result = self.directives['warn.unused_result']
        for assmt in assignments:
            if not assmt.refs:
                if assmt.entry.cf_references and warn_unused_result:
                    if assmt.is_arg:
                        self.messages.warning(
                            assmt, "Unused argument value '%s'" %
                                                assmt.entry.name)
                    else:
                        self.messages.warning(
                            assmt, "Unused result in '%s'" %
                                                assmt.entry.name)
                assmt.lhs.cf_used = False

    def warn_unreachable(self, node):
        "Generate a warning for unreachable code"
        if hasattr(node, 'lineno'):
            self.messages.warning(node, "Unreachable code")
