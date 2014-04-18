from __future__ import print_function, division, absolute_import
import dis
from collections import defaultdict
import functools
from llvm.core import Type, Builder, Module
import llvm.core as lc
from numba import ir, types, cgutils, utils, config


try:
    import builtins
except ImportError:
    import __builtin__ as builtins


class LoweringError(Exception):
    def __init__(self, msg, loc):
        self.msg = msg
        self.loc = loc
        super(LoweringError, self).__init__("%s\n%s" % (msg, loc.strformat()))


def default_mangler(name, argtypes):
    codedargs = '.'.join(str(a).replace(' ', '_') for a in argtypes)
    return '.'.join([name, codedargs])


class FunctionDescriptor(object):
    __slots__ = ('native', 'pymod', 'name', 'doc', 'blocks', 'typemap',
                 'calltypes', 'args', 'kws', 'restype', 'argtypes',
                 'qualified_name', 'mangled_name')

    def __init__(self, native, pymod, name, doc, blocks, typemap,
                 restype, calltypes, args, kws, mangler=None, argtypes=None,
                 qualname=None):
        self.native = native
        self.pymod = pymod
        self.name = name
        self.doc = doc
        self.blocks = blocks
        self.typemap = typemap
        self.calltypes = calltypes
        self.args = args
        self.kws = kws
        self.restype = restype
        # Argument types
        self.argtypes = argtypes or [self.typemap[a] for a in args]
        self.qualified_name = qualname or '.'.join([self.pymod.__name__,
                                                    self.name])
        mangler = default_mangler if mangler is None else mangler
        self.mangled_name = mangler(self.qualified_name, self.argtypes)


def _describe(interp):
    func = interp.bytecode.func
    fname = interp.bytecode.func_name
    pymod = interp.bytecode.module
    doc = func.__doc__ or ''
    args = interp.argspec.args
    kws = ()        # TODO
    return fname, pymod, doc, args, kws


def describe_external(name, restype, argtypes):
    args = ["arg%d" % i for i in range(len(argtypes))]
    fd = FunctionDescriptor(native=True, pymod=None, name=name, doc='',
                            blocks=None, restype=restype, calltypes=None,
                            argtypes=argtypes, args=args, kws=None,
                            typemap=None, qualname=name, mangler=lambda a, x: a)
    return fd


def describe_function(interp, typemap, restype, calltypes, mangler):
    fname, pymod, doc, args, kws = _describe(interp)
    native = True
    sortedblocks = utils.SortedMap(utils.dict_iteritems(interp.blocks))
    fd = FunctionDescriptor(native, pymod, fname, doc, sortedblocks,
                            typemap, restype, calltypes, args, kws, mangler)
    return fd


def describe_pyfunction(interp):
    fname, pymod, doc, args, kws = _describe(interp)
    defdict = lambda: defaultdict(lambda: types.pyobject)
    typemap = defdict()
    restype = types.pyobject
    calltypes = defdict()
    native = False
    sortedblocks = utils.SortedMap(utils.dict_iteritems(interp.blocks))
    fd = FunctionDescriptor(native, pymod, fname, doc, sortedblocks,
                            typemap, restype, calltypes, args, kws)
    return fd


class BaseLower(object):
    """
    Lower IR to LLVM
    """

    def __init__(self, context, fndesc):
        self.context = context
        self.fndesc = fndesc
        # Initialize LLVM
        self.module = Module.new("module.%s" % self.fndesc.name)

        # Install metadata
        md_pymod = cgutils.MetadataKeyStore(self.module, "python.module")
        md_pymod.set(fndesc.pymod.__name__)

        # Setup function
        self.function = context.declare_function(self.module, fndesc)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder.new(self.entry_block)
        # self.builder = cgutils.VerboseProxy(self.builder)

        # Internal states
        self.blkmap = {}
        self.varmap = {}
        self.firstblk = min(self.fndesc.blocks.keys())

        # Subclass initialization
        self.init()

    def init(self):
        pass

    def post_lower(self):
        """Called after all blocks are lowered
        """
        pass

    def lower(self):
        # Init argument variables
        fnargs = self.context.get_arguments(self.function)
        for ak, av in zip(self.fndesc.args, fnargs):
            at = self.typeof(ak)
            av = self.context.get_argument_value(self.builder, at, av)
            av = self.init_argument(av)
            self.storevar(av, ak)
            # Init blocks
        for offset in self.fndesc.blocks:
            bname = "B%d" % offset
            self.blkmap[offset] = self.function.append_basic_block(bname)
            # Lower all blocks
        for offset, block in self.fndesc.blocks.items():
            bb = self.blkmap[offset]
            self.builder.position_at_end(bb)
            self.lower_block(block)

        self.post_lower()
        # Close entry block
        self.builder.position_at_end(self.entry_block)
        self.builder.branch(self.blkmap[self.firstblk])

        if config.DUMP_LLVM:
            print(("LLVM DUMP %s" % self.fndesc.qualified_name).center(80, '-'))
            print(self.module)
            print('=' * 80)
        self.module.verify()

    def init_argument(self, arg):
        return arg

    def lower_block(self, block):
        for inst in block.body:
            try:
                self.lower_inst(inst)
            except LoweringError:
                raise
            except Exception as e:
                msg = "Internal error:\n%s: %s" % (type(e).__name__, e)
                raise LoweringError(msg, inst.loc)

    def typeof(self, varname):
        return self.fndesc.typemap[varname]


class Lower(BaseLower):
    def lower_inst(self, inst):
        if config.DEBUG_JIT:
            self.context.debug_print(self.builder, str(inst))
        if isinstance(inst, ir.Assign):
            ty = self.typeof(inst.target.name)
            val = self.lower_assign(ty, inst)
            self.storevar(val, inst.target.name)

        elif isinstance(inst, ir.Branch):
            cond = self.loadvar(inst.cond.name)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]

            condty = self.typeof(inst.cond.name)
            pred = self.context.cast(self.builder, cond, condty, types.boolean)
            assert pred.type == Type.int(1), ("cond is not i1: %s" % pred.type)
            self.builder.cbranch(pred, tr, fl)

        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)

        elif isinstance(inst, ir.Return):
            val = self.loadvar(inst.value.name)
            oty = self.typeof(inst.value.name)
            ty = self.fndesc.restype
            if isinstance(ty, types.Optional):
                if oty == types.none:
                    self.context.return_native_none(self.builder)
                    return
                else:
                    ty = ty.type

            if ty != oty:
                val = self.context.cast(self.builder, val, oty, ty)
            retval = self.context.get_return_value(self.builder, ty, val)
            self.context.return_value(self.builder, retval)

        elif isinstance(inst, ir.SetItem):
            target = self.loadvar(inst.target.name)
            value = self.loadvar(inst.value.name)
            index = self.loadvar(inst.index.name)

            targetty = self.typeof(inst.target.name)
            valuety = self.typeof(inst.value.name)
            indexty = self.typeof(inst.index.name)

            signature = self.fndesc.calltypes[inst]
            assert signature is not None
            impl = self.context.get_function('setitem', signature)

            # Convert argument to match
            assert targetty == signature.args[0]
            index = self.context.cast(self.builder, index, indexty,
                                      signature.args[1])
            value = self.context.cast(self.builder, value, valuety,
                                      signature.args[2])

            return impl(self.builder, (target, index, value))

        elif isinstance(inst, ir.Del):
            pass

        else:
            raise NotImplementedError(type(inst))

    def lower_assign(self, ty, inst):
        value = inst.value
        if isinstance(value, ir.Const):
            if self.context.is_struct_type(ty):
                const = self.context.get_constant_struct(self.builder, ty,
                                                         value.value)
            elif ty == types.string:
                const = self.context.get_constant_string(self.builder, ty,
                                                         value.value)
            else:
                const = self.context.get_constant(ty, value.value)
            return const

        elif isinstance(value, ir.Expr):
            return self.lower_expr(ty, value)

        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            oty = self.typeof(value.name)
            return self.context.cast(self.builder, val, oty, ty)

        elif isinstance(value, ir.Global):
            if (isinstance(ty, types.Dummy) or
                    isinstance(ty, types.Module) or
                    isinstance(ty, types.Function) or
                    isinstance(ty, types.Dispatcher)):
                return self.context.get_dummy_value()

            elif ty == types.boolean:
                return self.context.get_constant(ty, value.value)

            elif isinstance(ty, types.Array):
                return self.context.make_constant_array(self.builder, ty,
                                                        value.value)

            elif self.context.is_struct_type(ty):
                return self.context.get_constant_struct(self.builder, ty,
                                                        value.value)

            elif ty in types.number_domain:
                return self.context.get_constant(ty, value.value)

            elif isinstance(ty, types.UniTuple):
                consts = [self.context.get_constant(t, v)
                          for t, v in zip(ty, value.value)]
                return cgutils.pack_array(self.builder, consts)
            else:
                raise NotImplementedError('global', ty)

        else:
            raise NotImplementedError(type(value), value)

    def lower_expr(self, resty, expr):
        if expr.op == 'binop':
            lhs = expr.lhs
            rhs = expr.rhs
            lty = self.typeof(lhs.name)
            rty = self.typeof(rhs.name)
            lhs = self.loadvar(lhs.name)
            rhs = self.loadvar(rhs.name)
            # Get function
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.fn, signature)
            # Convert argument to match
            lhs = self.context.cast(self.builder, lhs, lty, signature.args[0])
            rhs = self.context.cast(self.builder, rhs, rty, signature.args[1])
            res = impl(self.builder, (lhs, rhs))
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == 'unary':
            val = self.loadvar(expr.value.name)
            typ = self.typeof(expr.value.name)
            # Get function
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.fn, signature)
            # Convert argument to match
            val = self.context.cast(self.builder, val, typ, signature.args[0])
            res = impl(self.builder, [val])
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == 'call':

            argvals = [self.loadvar(a.name) for a in expr.args]
            argtyps = [self.typeof(a.name) for a in expr.args]
            signature = self.fndesc.calltypes[expr]

            if isinstance(expr.func, ir.Intrinsic):
                fnty = expr.func.name
                castvals = expr.func.args
            else:
                assert not expr.kws, expr.kws
                fnty = self.typeof(expr.func.name)

                castvals = [self.context.cast(self.builder, av, at, ft)
                            for av, at, ft in zip(argvals, argtyps,
                                                  signature.args)]

            if isinstance(fnty, types.Method):
                # Method of objects are handled differently
                fnobj = self.loadvar(expr.func.name)
                res = self.context.call_class_method(self.builder, fnobj,
                                                     signature, castvals)

            elif isinstance(fnty, types.FunctionPointer):
                # Handle function pointer)
                pointer = fnty.funcptr
                res = self.context.call_function_pointer(self.builder, pointer,
                                                         signature, castvals)

            else:
                # Normal function resolution
                impl = self.context.get_function(fnty, signature)
                res = impl(self.builder, castvals)
                libs = getattr(impl, "libs", ())
                if libs:
                    self.context.add_libs(libs)
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op in ('getiter', 'iternext', 'itervalid', 'iternextsafe'):
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function(expr.op, signature)
            [fty] = signature.args
            castval = self.context.cast(self.builder, val, ty, fty)
            res = impl(self.builder, (castval,))
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == "getattr":
            val = self.loadvar(expr.value.name)
            ty = self.typeof(expr.value.name)
            impl = self.context.get_attribute(val, ty, expr.attr)
            if impl is None:
                # ignore the attribute
                res = self.context.get_dummy_value()
            else:
                res = impl(self.context, self.builder, ty, val)
                if not isinstance(impl.return_type, types.Kind):
                    res = self.context.cast(self.builder, res, impl.return_type,
                                            resty)
            return res

        elif expr.op == "getitem":
            baseval = self.loadvar(expr.target.name)
            indexval = self.loadvar(expr.index.name)
            signature = self.fndesc.calltypes[expr]
            impl = self.context.get_function("getitem", signature)
            argvals = (baseval, indexval)
            argtyps = (self.typeof(expr.target.name),
                       self.typeof(expr.index.name))
            castvals = [self.context.cast(self.builder, av, at, ft)
                        for av, at, ft in zip(argvals, argtyps,
                                              signature.args)]
            res = impl(self.builder, castvals)
            return self.context.cast(self.builder, res, signature.return_type,
                                     resty)

        elif expr.op == "build_tuple":
            itemvals = [self.loadvar(i.name) for i in expr.items]
            itemtys = [self.typeof(i.name) for i in expr.items]
            castvals = [self.context.cast(self.builder, val, fromty, toty)
                        for val, toty, fromty in zip(itemvals, resty, itemtys)]
            tup = self.context.get_constant_undef(resty)
            for i in range(len(castvals)):
                tup = self.builder.insert_value(tup, itemvals[i], i)
            return tup

        raise NotImplementedError(expr)

    def getvar(self, name):
        if name not in self.varmap:
            self.varmap[name] = self.alloca(name, self.typeof(name))
        return self.varmap[name]

    def loadvar(self, name):
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def storevar(self, value, name):
        ptr = self.getvar(name)
        assert value.type == ptr.type.pointee, \
            "store %s to ptr of %s" % (value.type, ptr.type.pointee)
        self.builder.store(value, ptr)

    def alloca(self, name, type):
        ltype = self.context.get_value_type(type)
        bb = self.builder.basic_block
        self.builder.position_at_end(self.entry_block)
        ptr = self.builder.alloca(ltype, name=name)
        self.builder.position_at_end(bb)
        return ptr


PYTHON_OPMAP = {
    '+': "number_add",
    '-': "number_subtract",
    '*': "number_multiply",
    '/?': "number_divide",
    '/': "number_truedivide",
    '//': "number_floordivide",
    '%': "number_remainder",
    '**': "number_power",
    '<<': "number_lshift",
    '>>': "number_rshift",
    '&': "number_and",
    '|': "number_or",
    '^': "number_xor",
}


class PyLower(BaseLower):
    def init(self):
        self.pyapi = self.context.get_python_api(self.builder)

        # Add error handling block
        self.ehblock = self.function.append_basic_block('error')

    def post_lower(self):
        with cgutils.goto_block(self.builder, self.ehblock):
            self.cleanup()
            self.context.return_exc(self.builder)

    def init_argument(self, arg):
        self.incref(arg)
        return arg

    def lower_inst(self, inst):
        if isinstance(inst, ir.Assign):
            value = self.lower_assign(inst)
            self.storevar(value, inst.target.name)

        elif isinstance(inst, ir.SetItem):
            target = self.loadvar(inst.target.name)
            index = self.loadvar(inst.index.name)
            value = self.loadvar(inst.value.name)
            ok = self.pyapi.object_setitem(target, index, value)
            negone = lc.Constant.int_signextend(ok.type, -1)
            pred = self.builder.icmp(lc.ICMP_EQ, ok, negone)
            with cgutils.if_unlikely(self.builder, pred):
                self.return_exception_raised()

        elif isinstance(inst, ir.Return):
            retval = self.loadvar(inst.value.name)
            self.incref(retval)
            self.cleanup()
            self.context.return_value(self.builder, retval)

        elif isinstance(inst, ir.Branch):
            cond = self.loadvar(inst.cond.name)
            if cond.type == Type.int(1):
                istrue = cond
            else:
                istrue = self.pyapi.object_istrue(cond)
            zero = lc.Constant.null(istrue.type)
            pred = self.builder.icmp(lc.ICMP_NE, istrue, zero)
            tr = self.blkmap[inst.truebr]
            fl = self.blkmap[inst.falsebr]
            self.builder.cbranch(pred, tr, fl)

        elif isinstance(inst, ir.Jump):
            target = self.blkmap[inst.target]
            self.builder.branch(target)

        elif isinstance(inst, ir.Del):
            obj = self.loadvar(inst.value)
            self.decref(obj)

        else:
            raise NotImplementedError(type(inst), inst)

    def lower_assign(self, inst):
        """
        The returned object must have a new reference
        """
        value = inst.value
        if isinstance(value, ir.Const):
            return self.lower_const(value.value)
        elif isinstance(value, ir.Var):
            val = self.loadvar(value.name)
            self.incref(val)
            return val
        elif isinstance(value, ir.Expr):
            return self.lower_expr(value)
        elif isinstance(value, ir.Global):
            return self.lower_global(value.name, value.value)
        else:
            raise NotImplementedError(type(value), value)

    def lower_expr(self, expr):
        if expr.op == 'binop':
            lhs = self.loadvar(expr.lhs.name)
            rhs = self.loadvar(expr.rhs.name)
            if expr.fn in PYTHON_OPMAP:
                fname = PYTHON_OPMAP[expr.fn]
                fn = getattr(self.pyapi, fname)
                res = fn(lhs, rhs)
            else:
                # Assume to be rich comparision
                res = self.pyapi.object_richcompare(lhs, rhs, expr.fn)
            self.check_error(res)
            return res
        elif expr.op == 'unary':
            value = self.loadvar(expr.value.name)
            if expr.fn == '-':
                res = self.pyapi.number_negative(value)
            elif expr.fn == 'not':
                res = self.pyapi.object_not(value)
                negone = lc.Constant.int_signextend(Type.int(), -1)
                err = self.builder.icmp(lc.ICMP_EQ, res, negone)
                with cgutils.if_unlikely(self.builder, err):
                    self.return_exception_raised()

                longval = self.builder.zext(res, self.pyapi.long)
                res = self.pyapi.bool_from_long(longval)
            elif expr.fn == '~':
                res = self.pyapi.number_invert(value)
            else:
                raise NotImplementedError(expr)
            self.check_error(res)
            return res
        elif expr.op == 'call':
            argvals = [self.loadvar(a.name) for a in expr.args]
            fn = self.loadvar(expr.func.name)
            if not expr.kws:
                # No keyword
                ret = self.pyapi.call_function_objargs(fn, argvals)
            else:
                # Have Keywords
                keyvalues = [(k, self.loadvar(v.name)) for k, v in expr.kws]
                args = self.pyapi.tuple_pack(argvals)
                kws = self.pyapi.dict_pack(keyvalues)
                ret = self.pyapi.call(fn, args, kws)
                self.decref(kws)
                self.decref(args)
            self.check_error(ret)
            return ret
        elif expr.op == 'getattr':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getattr_string(obj, expr.attr)
            self.check_error(res)
            return res
        elif expr.op == 'build_tuple':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.tuple_pack(items)
            self.check_error(res)
            return res
        elif expr.op == 'build_list':
            items = [self.loadvar(it.name) for it in expr.items]
            res = self.pyapi.list_pack(items)
            self.check_error(res)
            return res
        elif expr.op == 'getiter':
            obj = self.loadvar(expr.value.name)
            res = self.pyapi.object_getiter(obj)
            self.check_error(res)
            self.storevar(res, '$iter$' + expr.value.name)
            return self.pack_iter(res)
        elif expr.op == 'iternext':
            iterstate = self.loadvar(expr.value.name)
            iterobj, valid = self.unpack_iter(iterstate)
            item = self.pyapi.iter_next(iterobj)
            self.set_iter_valid(iterstate, item)
            return item
        elif expr.op == 'iternextsafe':
            iterstate = self.loadvar(expr.value.name)
            iterobj, _ = self.unpack_iter(iterstate)
            item = self.pyapi.iter_next(iterobj)
            # TODO need to add exception
            self.check_error(item)
            self.set_iter_valid(iterstate, item)
            return item
        elif expr.op == 'itervalid':
            iterstate = self.loadvar(expr.value.name)
            _, valid = self.unpack_iter(iterstate)
            return self.builder.trunc(valid, Type.int(1))
        elif expr.op == 'getitem':
            target = self.loadvar(expr.target.name)
            index = self.loadvar(expr.index.name)
            res = self.pyapi.object_getitem(target, index)
            self.check_error(res)
            return res
        elif expr.op == 'getslice':
            target = self.loadvar(expr.target.name)
            start = self.loadvar(expr.start.name)
            stop = self.loadvar(expr.stop.name)

            slicefn = self.get_builtin_obj("slice")
            sliceobj = self.pyapi.call_function_objargs(slicefn, (start, stop))
            self.decref(slicefn)
            self.check_error(sliceobj)

            res = self.pyapi.object_getitem(target, sliceobj)
            self.check_error(res)

            return res
        else:
            raise NotImplementedError(expr)

    def lower_const(self, const):
        if isinstance(const, str):
            ret = self.pyapi.string_from_string_and_size(const)
            self.check_error(ret)
            return ret
        elif isinstance(const, complex):
            real = self.context.get_constant(types.float64, const.real)
            imag = self.context.get_constant(types.float64, const.imag)
            ret = self.pyapi.complex_from_doubles(real, imag)
            self.check_error(ret)
            return ret
        elif isinstance(const, float):
            fval = self.context.get_constant(types.float64, const)
            ret = self.pyapi.float_from_double(fval)
            self.check_error(ret)
            return ret
        elif isinstance(const, int):
            if utils.bit_length(const) >= 64:
                raise ValueError("Integer is too big to be lowered")
            ival = self.context.get_constant(types.intp, const)
            return self.pyapi.long_from_ssize_t(ival)
        elif isinstance(const, tuple):
            items = [self.lower_const(i) for i in const]
            return self.pyapi.tuple_pack(items)
        elif const is Ellipsis:
            return self.get_builtin_obj("Ellipsis")
        elif const is None:
            return self.pyapi.make_none()
        else:
            raise NotImplementedError(type(const))

    def lower_global(self, name, value):
        """
        1) Check global scope dictionary.
        2) Check __builtins__.
            2a) is it a dictionary (for non __main__ module)
            2b) is it a module (for __main__ module)
        """
        moddict = self.pyapi.get_module_dict()
        obj = self.pyapi.dict_getitem_string(moddict, name)
        self.incref(obj)  # obj is borrowed

        if hasattr(builtins, name):
            obj_is_null = self.is_null(obj)
            bbelse = self.builder.basic_block

            with cgutils.ifthen(self.builder, obj_is_null):
                mod = self.pyapi.dict_getitem_string(moddict, "__builtins__")
                builtin = self.builtin_lookup(mod, name)
                bbif = self.builder.basic_block

            retval = self.builder.phi(self.pyapi.pyobj)
            retval.add_incoming(obj, bbelse)
            retval.add_incoming(builtin, bbif)

        else:
            retval = obj
            with cgutils.if_unlikely(self.builder, self.is_null(retval)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()

        self.incref(retval)
        return retval

    # -------------------------------------------------------------------------

    def get_builtin_obj(self, name):
        moddict = self.pyapi.get_module_dict()
        mod = self.pyapi.dict_getitem_string(moddict, "__builtins__")
        return self.builtin_lookup(mod, name)

    def builtin_lookup(self, mod, name):
        """
        Args
        ----
        mod:
            The __builtins__ dictionary or module
        name: str
            The object to lookup
        """
        fromdict = self.pyapi.dict_getitem_string(mod, name)
        self.incref(fromdict)       # fromdict is borrowed
        bbifdict = self.builder.basic_block

        with cgutils.if_unlikely(self.builder, self.is_null(fromdict)):
            # This happen if we are using the __main__ module
            frommod = self.pyapi.object_getattr_string(mod, name)

            with cgutils.if_unlikely(self.builder, self.is_null(frommod)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()

            bbifmod = self.builder.basic_block

        builtin = self.builder.phi(self.pyapi.pyobj)
        builtin.add_incoming(fromdict, bbifdict)
        builtin.add_incoming(frommod, bbifmod)

        return builtin

    def pack_iter(self, obj):
        iterstate = PyIterState(self.context, self.builder)
        iterstate.iterator = obj
        iterstate.valid = cgutils.true_byte
        return iterstate._getpointer()

    def unpack_iter(self, state):
        iterstate = PyIterState(self.context, self.builder, ref=state)
        return tuple(iterstate)

    def set_iter_valid(self, state, item):
        iterstate = PyIterState(self.context, self.builder, ref=state)
        iterstate.valid = cgutils.as_bool_byte(self.builder,
                                               cgutils.is_not_null(self.builder,
                                                                   item))

        with cgutils.if_unlikely(self.builder, self.is_null(item)):
            self.check_occurred()

    def check_occurred(self):
        err_occurred = cgutils.is_not_null(self.builder,
                                           self.pyapi.err_occurred())

        with cgutils.if_unlikely(self.builder, err_occurred):
            self.return_exception_raised()

    def check_error(self, obj):
        with cgutils.if_unlikely(self.builder, self.is_null(obj)):
            self.return_exception_raised()

        return obj

    def is_null(self, obj):
        return cgutils.is_null(self.builder, obj)

    def return_exception_raised(self):
        self.builder.branch(self.ehblock)

    def return_error_occurred(self):
        self.cleanup()
        self.context.return_exc(self.builder)

    def getvar(self, name, ltype=None):
        if name not in self.varmap:
            self.varmap[name] = self.alloca(name, ltype=ltype)
        return self.varmap[name]

    def loadvar(self, name):
        ptr = self.getvar(name)
        return self.builder.load(ptr)

    def storevar(self, value, name):
        """
        Stores a llvm value and allocate stack slot if necessary.
        The llvm value can be of arbitrary type.
        """
        ptr = self.getvar(name, ltype=value.type)
        old = self.builder.load(ptr)
        assert value.type == ptr.type.pointee, (str(value.type),
                                                str(ptr.type.pointee))
        self.builder.store(value, ptr)
        # Safe to call decref even on non python object
        self.decref(old)

    def cleanup(self):
        for var in utils.dict_itervalues(self.varmap):
            self.decref(self.builder.load(var))

    def alloca(self, name, ltype=None):
        """
        Allocate a stack slot and initialize it to NULL.
        The default is to allocate a pyobject pointer.
        Use ``ltype`` to override.
        """
        if ltype is None:
            ltype = self.context.get_value_type(types.pyobject)
        bb = self.builder.basic_block
        self.builder.position_at_end(self.entry_block)
        ptr = self.builder.alloca(ltype, name=name)
        self.builder.store(cgutils.get_null_value(ltype), ptr)
        self.builder.position_at_end(bb)
        return ptr

    def incref(self, value):
        self.pyapi.incref(value)

    def decref(self, value):
        """
        This is allow to be called on non pyobject pointer, in which case
        no code is inserted.

        If the value is a PyIterState, it unpack the structure and decref
        the iterator.
        """
        lpyobj = self.context.get_value_type(types.pyobject)

        if value.type.kind == lc.TYPE_POINTER:
            if value.type != lpyobj:
                pass
                #raise AssertionError(value.type)
                # # Handle PyIterState
                # not_null = cgutils.is_not_null(self.builder, value)
                # with cgutils.if_likely(self.builder, not_null):
                #     iterstate = PyIterState(self.context, self.builder,
                #                             value=value)
                #     value = iterstate.iterator
                #     self.pyapi.decref(value)
            else:
                self.pyapi.decref(value)


class PyIterState(cgutils.Structure):
    _fields = [
        ("iterator", types.pyobject),
        ("valid", types.boolean),
    ]


def pure(fn):
    @functools.wraps(fn)
    def closure(self, inst):
        key = inst.opname
        if key not in self.block_cache:
            bb = self.builder.basic_block
            fn(self, inst)
            self.block_cache[key] = bb
        else:
            bb = self.block_cache[key]
            self.builder.branch(bb)

    return closure


def pure_by_arg(fn):
    @functools.wraps(fn)
    def closure(self, inst):
        key = inst.opname, inst.arg
        if key not in self.block_cache:
            bb = self.builder.basic_block
            fn(self, inst)
            self.block_cache[key] = bb
        else:
            bb = self.block_cache[key]
            self.builder.branch(bb)

    return closure


class PyLowerMiniInterp(object):
    def __init__(self, context, interp, fndesc):
        from numba import interpreter

        assert isinstance(interp, interpreter.Interpreter)
        self.context = context
        self.interp = interp
        self.fndesc = fndesc
        self.bytecode = interp.bytecode
        print(self.bytecode.dump())

        self.module = Module.new("module.%s" % self.fndesc.name)

        # Install metadata
        md_pymod = cgutils.MetadataKeyStore(self.module, "python.module")
        md_pymod.set(fndesc.pymod.__name__)

        # Setup function
        self.function = context.declare_function(self.module, fndesc)
        self.entry_block = self.function.append_basic_block('entry')
        self.builder = Builder.new(self.entry_block)

        self._ip = self.builder.alloca(Type.int(), name="ip")
        self.ip = 0

        self.instructions = list(self.bytecode)

        self.switch_block = self.function.append_basic_block('SWITCH')

        self.handler_blocks = [self.function.append_basic_block("OP_%d" % i)
                               for i in range(len(self.instructions))]
        self.unhandled_block = self.function.append_basic_block("OP_BAD")

        self.exit_block = self.function.append_basic_block("EXIT")

        # Init
        self.pyapi = self.context.get_python_api(self.builder)
        self.global_cache = {}
        self.block_cache = {}

        pyobj = self.pyapi.pyobj

        # Initialize interpreter states
        nvars = lc.Constant.int(Type.int(), len(self.bytecode.co_varnames))
        self.localvars = self.builder.alloca(pyobj, size=nvars, name="vars")
        self.sp = self.builder.alloca(Type.int(), name="sp")
        self.builder.store(lc.Constant.int(self.sp.type.pointee, 0), self.sp)
        # TODO handle stack size allocation correctly
        self.stacksize = lc.Constant.int(Type.int(), 100)
        self.stack = self.builder.alloca(pyobj, size=self.stacksize,
                                         name="stack")
        self.return_value = self.builder.alloca(pyobj, name='retval')

        # Set all locals to NULL
        null = lc.Constant.null(pyobj)
        for var in self.bytecode.co_varnames:
            self.storevar(null, var, gc=False)

        # Store arguments into locals
        for arg, argval in zip(self.interp.argspec.args,
                               self.context.get_arguments(self.function)):
            self.incref(argval)
            self.storevar(argval, arg, gc=False)

    def lower(self):
        self.builder.branch(self.switch_block)
        self.builder.position_at_end(self.switch_block)

        # Opcode switch
        swt = self.builder.switch(self.ip, self.unhandled_block,
                                  n=len(self.instructions))
        for i in range(len(self.instructions)):
            swt.add_case(lc.Constant.int(Type.int(), i), self.handler_blocks[i])

        # Unhandled opcode
        self.builder.position_at_end(self.unhandled_block)
        self.pyapi.raise_native_error("bad opcode")
        self.context.return_exc(self.builder)

        # Handle opcode
        for i, inst in enumerate(self.instructions):
            self.builder.position_at_end(self.handler_blocks[i])
            self._lower(inst)

        # Exit block
        self.builder.position_at_end(self.exit_block)
        self.epilog()
        retval = self.builder.load(self.return_value)
        with cgutils.ifelse(self.builder, self.is_null(retval)) as \
            (is_exc, is_ret):
            with is_exc:
                self.context.return_exc(self.builder)

            with is_ret:
                self.context.return_value(self.builder, retval)

        self.builder.unreachable()

    def _lower(self, inst):
        fn = getattr(self, "lower_%s" % inst.opname)
        fn(inst)

    @pure_by_arg
    def lower_LOAD_CONST(self, inst):
        const = self.bytecode.co_consts[inst.arg]
        if isinstance(const, int):
            lconst = lc.Constant.int_signextend(self.pyapi.py_ssize_t,
                                                const)
            self.push(self.pyapi.long_from_ssize_t(lconst))
            self.passthrough()
        elif isinstance(const, str):
            val = self.pyapi.string_from_string_and_size(const)
            self.push(val)
            self.passthrough()
        else:
            raise NotImplementedError(inst, const, type(const))

    @pure
    def lower_POP_TOP(self, inst):
        tos = self.tos()
        self.incref(tos)
        self.push(tos)
        self.passthrough()

    @pure
    def lower_DUP_TOP(self, inst):
        val = self.pop()
        self.decref(val)
        self.passthrough()

    @pure_by_arg
    def lower_LOAD_FAST(self, inst):
        name = self.bytecode.co_varnames[inst.arg]
        val = self.loadvar(name)
        self.incref(val)
        self.push(val)
        self.passthrough()

    @pure_by_arg
    def lower_STORE_FAST(self, inst):
        name = self.bytecode.co_varnames[inst.arg]
        val = self.pop()
        self.storevar(val, name)
        self.passthrough()

    @pure_by_arg
    def lower_LOAD_GLOBAL(self, inst):
        name = self.bytecode.co_names[inst.arg]
        val = self.load_global_once(name)
        self.incref(val)
        self.push(val)
        self.passthrough()

    @pure_by_arg
    def lower_LOAD_ATTR(self, inst):
        attr = self.bytecode.co_names[inst.arg]
        base = self.pop()
        res = self.pyapi.object_getattr_string(base, attr)
        self.decref(base)
        self.check_error(res)
        self.push(res)

    @pure
    def lower_SETUP_LOOP(self, inst):
        self.passthrough()

    @pure
    def lower_POP_BLOCK(self, inst):
        self.passthrough()

    def binary_op(self, fn):
        rhs = self.pop()
        lhs = self.pop()
        result = fn(lhs, rhs)
        self.decref(lhs)
        self.decref(rhs)
        self.check_error(result)
        self.push(result)
        self.passthrough()

    @pure
    def lower_BINARY_ADD(self, inst):
        self.binary_op(self.pyapi.number_add)

    @pure
    def lower_BINARY_SUBTRACT(self, inst):
        self.binary_op(self.pyapi.number_subtract)

    @pure
    def lower_BINARY_MULTIPLY(self, inst):
        self.binary_op(self.pyapi.number_multiply)

    @pure
    def lower_BINARY_DIVIDE(self, inst):
        self.binary_op(self.pyapi.number_divide)

    @pure
    def lower_BINARY_TRUE_DIVIDE(self, inst):
        self.binary_op(self.pyapi.number_truedivide)

    @pure
    def lower_BINARY_FLOOR_DIVIDE(self, inst):
        self.binary_op(self.pyapi.number_floordivide)

    @pure
    def lower_BINARY_MODULO(self, inst):
        self.binary_op(self.pyapi.number_modulo)

    @pure
    def lower_BINARY_POWER(self, inst):
        self.binary_op(self.pyapi.number_power)

    @pure
    def lower_BINARY_LSHIFT(self, inst):
        self.binary_op(self.pyapi.number_lshift)

    @pure
    def lower_BINARY_RSHIFT(self, inst):
        self.binary_op(self.pyapi.number_rshift)

    @pure
    def lower_BINARY_AND(self, inst):
        self.binary_op(self.pyapi.number_and)

    @pure
    def lower_BINARY_OR(self, inst):
        self.binary_op(self.pyapi.number_or)

    @pure
    def lower_BINARY_XOR(self, inst):
        self.binary_op(self.pyapi.number_xor)

    lower_INPLACE_ADD = lower_BINARY_ADD
    lower_INPLACE_SUBTRACT = lower_BINARY_SUBTRACT
    lower_INPLACE_MULTIPLY = lower_BINARY_MULTIPLY
    lower_INPLACE_DIVIDE = lower_BINARY_DIVIDE
    lower_INPLACE_TRUE_DIVIDE = lower_BINARY_TRUE_DIVIDE
    lower_INPLACE_FLOOR_DIVIDE = lower_BINARY_FLOOR_DIVIDE
    lower_INPLACE_MODULO = lower_BINARY_MODULO
    lower_INPLACE_POWER = lower_BINARY_POWER
    lower_INPLACE_LSHIFT = lower_BINARY_LSHIFT
    lower_INPLACE_RSHIFT = lower_BINARY_RSHIFT
    lower_INPLACE_AND = lower_BINARY_AND
    lower_INPLACE_OR = lower_BINARY_OR
    lower_INPLACE_XOR = lower_BINARY_XOR

    def unary_op(self, fn):
        val = self.pop()
        res = fn(val)
        self.decref(val)
        self.check_error(res)
        self.push(res)

    @pure
    def lower_UNARY_NEGATIVE(self, inst):
        self.unary_op(self.pyapi.number_negative)

    @pure
    def lower_UNARY_NOT(self, inst):
        self.unary_op(self.pyapi.number_not)

    @pure
    def lower_UNARY_INVERT(self, inst):
        self.unary_op(self.pyapi.number_invert)

    @pure_by_arg
    def lower_COMPARE_OP(self, inst):
        rhs = self.pop()
        lhs = self.pop()
        opstr = dis.cmp_op[inst.arg]
        res = self.object_richcompare(lhs, rhs, opstr)
        self.decref(lhs)
        self.decref(rhs)
        self.check_error(res)
        self.push(res)

    @pure
    def lower_RETURN_VALUE(self, inst):
        val = self.pop()
        self.exit(val)

    @pure_by_arg
    def lower_CALL_FUNCTION(self, inst):
        narg = inst.arg & 0xff
        nkws = (inst.arg >> 8) & 0xff

        def pop_kws():
            val = self.pop()
            key = self.pop()
            return key, val

        kws = list(reversed([pop_kws() for _ in range(nkws)]))
        args = list(reversed([self.pop() for _ in range(narg)]))
        func = self.pop()

        if kws:
            raise NotImplementedError
        else:
            retval = self.pyapi.call_function_objargs(func, args)
            # clean up
        for a in args:
            self.decref(a)
        for k, v in kws:
            self.decref(k)
            self.decref(v)
        self.decref(func)

        self.check_error(retval)
        self.push(retval)
        self.passthrough()

    @pure
    def lower_GET_ITER(self, inst):
        val = self.pop()
        res = self.pyapi.object_getiter(val)
        self.decref(val)
        self.check_error(res)
        self.push(res)
        self.passthrough()

    def lower_FOR_ITER(self, inst):
        itobj = self.tos()
        nextval = self.pyapi.iter_next(itobj)
        with cgutils.ifelse(self.builder, self.is_null(nextval)) as (then,
                                                                     alt):
            with then:
                self.decref(self.pop())
                self.jump(inst.get_jump_target())

            with alt:
                self.push(nextval)
                self.passthrough()

        self.builder.unreachable()

    def lower_JUMP_ABSOLUTE(self, inst):
        self.jump(inst.get_jump_target())

    def jump(self, bytecode_offset):
        for i, inst in enumerate(self.instructions):
            if inst.offset == bytecode_offset:
                self.goto(i)
                return
        raise AssertionError("unreachable")

    def passthrough(self):
        self.goto(self.builder.add(self.ip, lc.Constant.int(Type.int(), 1)))

    def clear_locals(self):
        for name in self.bytecode.co_varnames:
            self.decref(self.loadvar(name))

    @property
    def ip(self):
        return self.builder.load(self._ip)

    @ip.setter
    def ip(self, ip):
        if isinstance(ip, int):
            ip = lc.Constant.int(Type.int(), ip)
        self.builder.store(ip, self._ip)

    def goto(self, ip):
        ## Note: Direct Jump? Does not seem to have much effect.
        # if isinstance(ip, int):
        #     self.ip = ip
        #     self.builder.branch(self.handler_blocks[ip])
        # else:
        self.ip = ip
        self.builder.branch(self.switch_block)

    def check_error(self, val):
        is_null = cgutils.is_null(self.builder, val)
        with cgutils.if_unlikely(self.builder, is_null):
            self.raises()

    def storevar(self, value, name, gc=True):
        offset = self.bytecode.co_varnames.index(name)
        offset = lc.Constant.int(Type.int(), offset)
        ptr = self.builder.gep(self.localvars, [offset])
        orig = self.builder.load(ptr)
        self.builder.store(value, ptr)
        if gc:
            # If the old value is not null, decref
            is_not_null = cgutils.is_not_null(self.builder, orig)
            with cgutils.ifthen(self.builder, is_not_null):
                self.decref(orig)

    def loadvar(self, name):
        offset = self.bytecode.co_varnames.index(name)
        offset = lc.Constant.int(Type.int(), offset)
        ptr = self.builder.gep(self.localvars, [offset])
        return self.builder.load(ptr)

    def tos(self):
        sp = self.builder.load(self.sp)
        nsp = self.builder.sub(sp, lc.Constant.int(Type.int(), 1))
        ptr = self.builder.gep(self.stack, [nsp])
        val = self.builder.load(ptr)
        return val

    def push(self, val):
        sp = self.builder.load(self.sp)
        ptr = self.builder.gep(self.stack, [sp])
        self.builder.store(val, ptr)
        nsp = self.builder.add(sp, lc.Constant.int(Type.int(), 1))
        self.builder.store(nsp, self.sp)

    def pop(self):
        sp = self.builder.load(self.sp)
        nsp = self.builder.sub(sp, lc.Constant.int(Type.int(), 1))
        ptr = self.builder.gep(self.stack, [nsp])
        val = self.builder.load(ptr)
        self.builder.store(nsp, self.sp)
        return val

    def unwind(self):
        sp = self.builder.load(self.sp)
        with cgutils.for_range(self.builder, sp, sp.type) as idx:
            ptr = self.builder.gep(self.stack, [idx])
            val = self.builder.load(ptr)
            self.decref(val)

    def incref(self, value):
        self.pyapi.incref(value)

    def decref(self, value):
        self.pyapi.decref(value)

    def load_global_once(self, name):
        if name not in self.global_cache:
            self.global_cache[name] = gv = self.module.add_global_variable(
                self.pyapi.pyobj, ".global.%s" % name)
            gv.initializer = lc.Constant.null(gv.type.pointee)
            gv.linkage = lc.LINKAGE_INTERNAL

        gv = self.global_cache[name]

        val = self.builder.load(gv)
        with cgutils.if_unlikely(self.builder, self.is_null(val)):
            val = self.load_global(name)
            self.builder.store(val, gv)

        return self.builder.load(gv)

    def load_global(self, name):
        """
        1) Check global scope dictionary.
        2) Check __builtins__.
            2a) is it a dictionary (for non __main__ module)
            2b) is it a module (for __main__ module)
        """
        moddict = self.pyapi.get_module_dict()
        obj = self.pyapi.dict_getitem_string(moddict, name)
        self.incref(obj)  # obj is borrowed

        if hasattr(builtins, name):
            obj_is_null = self.is_null(obj)
            bbelse = self.builder.basic_block

            with cgutils.ifthen(self.builder, obj_is_null):
                mod = self.pyapi.dict_getitem_string(moddict, "__builtins__")
                builtin = self.builtin_lookup(mod, name)
                bbif = self.builder.basic_block

            retval = self.builder.phi(self.pyapi.pyobj)
            retval.add_incoming(obj, bbelse)
            retval.add_incoming(builtin, bbif)

        else:
            retval = obj
            with cgutils.if_unlikely(self.builder, self.is_null(retval)):
                self.pyapi.raise_missing_global_error(name)
                self.raises()

        self.incref(retval)
        return retval

    def get_builtin_obj(self, name):
        moddict = self.pyapi.get_module_dict()
        mod = self.pyapi.dict_getitem_string(moddict, "__builtins__")
        return self.builtin_lookup(mod, name)

    def builtin_lookup(self, mod, name):
        """
        Args
        ----
        mod:
            The __builtins__ dictionary or module
        name: str
            The object to lookup
        """
        fromdict = self.pyapi.dict_getitem_string(mod, name)
        self.incref(fromdict)       # fromdict is borrowed
        bbifdict = self.builder.basic_block

        with cgutils.if_unlikely(self.builder, self.is_null(fromdict)):
            # This happen if we are using the __main__ module
            frommod = self.pyapi.object_getattr_string(mod, name)

            with cgutils.if_unlikely(self.builder, self.is_null(frommod)):
                self.pyapi.raise_missing_global_error(name)
                self.return_exception_raised()

            bbifmod = self.builder.basic_block

        builtin = self.builder.phi(self.pyapi.pyobj)
        builtin.add_incoming(fromdict, bbifdict)
        builtin.add_incoming(frommod, bbifmod)

        return builtin

    def return_exception_raised(self):
        self.clear_locals()
        self.context.return_exc(self.builder)

    def is_null(self, val):
        return cgutils.is_null(self.builder, val)

    def epilog(self):
        self.clear_locals()
        self.unwind()

    def exit(self, retval):
        self.builder.store(retval, self.return_value)
        self.builder.branch(self.exit_block)

    def raises(self):
        self.builder.store(lc.Constant.null(self.return_value.type.pointee),
                           self.return_value)
        self.builder.branch(self.exit_block)
