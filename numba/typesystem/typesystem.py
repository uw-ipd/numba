from __future__ import print_function, absolute_import
from numba.typesystem import _typesystem


def _prepare_versions(vers):
    fvers = []
    for ver in vers:
        fvers.extend(ver)
    return fvers


def _unwrap_type_list(tl):
    return [t.handle for t in tl]


class TypeContext(object):
    def __init__(self):
        self.handle = _typesystem.new_typecontext()
        _typesystem.fill_machine_types(self.handle)

    def get_type(self, name):
        """Get a type by name
        """
        hh = _typesystem.get_type(self.handle, name)
        return Type(self, hh)

    def resolve(self, sig, vers):
        """Apply overload resolution
        """
        fsig = _unwrap_type_list(sig)
        fvers = _unwrap_type_list(_prepare_versions(vers))
        sels = _typesystem.select_overload(self.handle, fsig, fvers)
        return tuple(vers[i] for i in sels)

    def best_resolve(self, sig, vers):
        """Apply overload resolution and use asymmetric rule to resolve
        any ambiguity.
        """
        fsig = _unwrap_type_list(sig)
        fvers = _unwrap_type_list(_prepare_versions(vers))
        which = _typesystem.select_best_overload(self.handle, fsig, fvers)
        if which is not None:
            return vers[which]

    def cast(self, orig, dest):
        kind, dist = _typesystem.cast(self.handle, orig.handle,
                                      dest.handle)
        return CastDesc(kind, dist)

    def coerce(self, *types):
        types = _unwrap_type_list(types)
        finalty, safe = _typesystem.coerce(self.handle, types)
        return Type(self, finalty), safe


class CastDesc(object):
    cast_kinds = 'exact', 'promote', 'convert', 'false'

    def __init__(self, kind, dist):
        assert kind in self.cast_kinds
        self.kind = kind
        self.distance = dist

    def __cmp__(self, other):
        k1 = self.cast_kinds.index(self.kind)
        k2 = self.cast_kinds.index(other.kind)
        if k1 == k2:
            return self.dist - other.dist
        else:
            return k1 - k2

    def __bool__(self):
        return self.kind != 'false'

    def __repr__(self):
        if self.is_coerce:
            return "<cast by=%s distance=%d>" % (self.kind, self.dist)
        else:
            return "<cast by=%s>" % (self.kind,)

    @property
    def is_coerce(self):
        return self.kind == "convert"

    @property
    def is_exact(self):
        return self.kind == "exact"

    @property
    def is_promote(self):
        return self.kind == "promote"


class Type(object):
    def __init__(self, context, handle):
        self.context = context
        self.handle = handle
        self.name = _typesystem.get_type_name(self.handle)

    def __repr__(self):
        return "<type %s>" % (self.name,)

    def __str__(self):
        return self.name

    def cast(self, to):
        return self.context.cast(self, to)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.name == other.name

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.name)
