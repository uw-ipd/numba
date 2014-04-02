from __future__ import print_function, absolute_import, division
import functools
import weakref
import numpy


class State(object):
    """A state of a value.
    It can be deferred, that it contains a transformation from other states.
    It can be manifested, that it contains the value.

    A deferred state can also be owned/free.
    The transformation in a deferred-free-state that is not owned can be
    composed with other deferred-free-states.  So that temporary
    sub-expressions can be eliminated, while the content
    of deferred-owned-states are manifested into usable concrete form.
    A manifested state must be owned.
    """

    @staticmethod
    def deferred(trans):
        return State(trans=trans)

    @staticmethod
    def manifested(value):
        return State(value=value)

    def __init__(self, trans=None, value=None):
        assert trans is None or value is None
        assert not (trans is None and value is None)
        self.trans = trans
        self.value = value
        self.owners = set()

    def force(self):
        """Force evaluation of the value if it is deferred.
        No-op if it is manifested.

        Returns
        -------
        manifested value
        """
        if self.is_deferred:
            self.value = self.trans.apply()
            self.trans = None
        assert self.is_manifested
        return self.value

    @property
    def is_deferred(self):
        return self.value is None

    @property
    def is_manifested(self):
        return not self.is_deferred

    @property
    def is_owned(self):
        return bool(self.owners)

    def __repr__(self):
        if self.is_manifested:
            return "<manifested %s>" % (type(self.value),)
        else:
            owned = "owned" if self.is_owned else "free"
            return "<deferred %s %r>" % (owned, self.trans)


class Transformation(object):
    def __init__(self, *args):
        self.args = args

    def __repr__(self):
        return "<%s nargs=%d>" % (type(self).__name__, len(self.args))

    def apply(self):
        """Subtype should implement this method.
        """
        return NotImplementedError


class AddOp(Transformation):
    def apply(self):
        left, right = self.args
        return left.force() + right.force()


class SubOp(Transformation):
    def apply(self):
        left, right = self.args
        return left.force() - right.force()


def binop(trans):
    """Decorator for binary operators to bind to a transformation class.
    """
    def wrap(fn):
        @functools.wraps(fn)
        def impl(self, other):
            state = State.deferred(trans(self.state, other.state))
            return Array(state=state)

        return impl

    return wrap


class Array(object):
    """A deferred array object.
    """
    def __init__(self, state):
        self.state = state
        self._ref = weakref.ref(self)
        self.state.owners.add(self._ref)

    def __del__(self):
        self.state.owners.discard(self._ref)

    @binop(AddOp)
    def __add__(self, other):
        pass

    @binop(SubOp)
    def __sub__(self, other):
        pass

    def force(self):
        self.state.force()

    def __repr__(self):
        return "<Array %s>" % self.state

    def __str__(self):
        self.force()
        return str(self.state.value)


def describe_state(state, _p="", _suppress=False):
    if not _suppress:
        print("Describe state dependencies")
    print(_p, '-', state)
    if state.is_deferred:
        for arg in state.trans.args:
            describe_state(arg, _p=_p + " |", _suppress=True)


def from_numpy(arr):
    return Array(state=State.manifested(arr))


# -------------
# Test

def test():
    print("# Build Array")
    arr = from_numpy(numpy.arange(10))
    describe_state(arr.state)

    print("# Build Expr")
    sum = arr + arr + arr
    describe_state(sum.state)


    print("# Force")
    print(sum)
    describe_state(sum.state)


if __name__ == '__main__':
    test()
