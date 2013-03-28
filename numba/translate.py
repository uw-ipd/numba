import dis, inspect
from io import BytesIO
from collections import namedtuple, deque, defaultdict
from llvmpy.api import llvm
from .bytecode_info import opcode_info_table
from . import llvm_passes, llvm_utils, llvm_types, llvm_const
from .llvm_utils import llvm_context
from .mangling import mangle
from .global_mappings import builtins_table

Instr = namedtuple('Instr', ['opname', 'operand', 'offset', 'length'])

def get_code_object(obj):
    "Shamelessly borrowed from llpython"
    return getattr(obj, '__code__', getattr(obj, 'func_code', None))

def gen_instr(co_code):
    iter_bytecode = iter((i, ord(x)) for i, x in enumerate(co_code)).next
    while True:
        offset, opcode = iter_bytecode()
        try:
            opinfo = opcode_info_table[opcode]
        except KeyError:
            raise KeyError(dis.opname[opcode])

        if opinfo.oplen:
            operand = 0
            for i in range(opinfo.oplen):
                operand |= iter_bytecode()[1] << (i * 8)
        else:
            operand = None
        yield Instr(opname=opinfo.opname, operand=operand, offset=offset,
                    length=opinfo.oplen + 1)

class Translator(object):
    def __init__(self, func):
        self.globals = func.func_globals
        self.builtins = vars(self.globals['__builtins__'])
        self.code = get_code_object(func)
        dis.disassemble(self.code)
        argspec = inspect.getargspec(func)
        self.argnames = argspec.args
        assert not argspec.varargs
        assert not argspec.keywords
        assert not argspec.defaults

    def execute(self):
        self.stack = deque()
        self.scope = deque()
        self.symtab = {}

        # make module
        self.lmod = llvm.Module.new('module.%s' % self.code.co_name,
                                    llvm_context)
        # make function
        argtys = [llvm_types.unknown_ptr] * len(self.argnames)
        is_vararg = False
        self.lfunc = llvm_utils.get_or_insert_func(self.lmod, self.code.co_name,
                                                   llvm_types.unknown_ptr,
                                                   argtys)
        assert self.lfunc.isDeclaration(), "Function already defined"

        # make basicblocks
        entry = self.make_block()
        labels = dis.findlabels(self.code.co_code)
        self.blocks = {}
        if not labels or 0 not in labels:
            self.blocks[0] = self.make_block(0)
        self.blocks.update(dict((label, self.make_block(label))
                                 for label in sorted(labels)))

        # make builder
        self.builder = llvm.IRBuilder.new(llvm_context)
        self.builder.SetInsertPoint(entry)

        # allocate stack variables
        for name in self.code.co_varnames:
            ptr = self.builder.CreateAlloca(llvm_types.unknown_ptr)
            self.symtab[name] = ptr

        # null initialize stack variables
        null = llvm_const.null(llvm_types.unknown_ptr)
        for name in self.code.co_varnames:
            ptr = self.symtab[name]
            self.builder.CreateStore(null, ptr)

        # name all arguments
        for name, arg in zip(self.argnames, self.lfunc.getArgumentList()):
            arg.setName(name)
            self.builder.CreateStore(arg, self.symtab[name])
            self.incref(arg)

        # interpret bytecode
        for instr in gen_instr(self.code.co_code):
            self.cur_instr = instr
            if self.cur_instr.offset in self.blocks:
                if self.builder.GetInsertBlock().getTerminator() is None:
                    self.builder.CreateBr(self.blocks[self.cur_instr.offset])
                self.builder.SetInsertPoint(self.blocks[self.cur_instr.offset])
            if self.builder.GetInsertBlock().getTerminator() is None:
                # if block is not terminated
                callee = getattr(self, 'op_%s' % instr.opname)
                if instr.operand is not None:
                    callee(instr.operand)
                else:
                    callee()
            
        else:
            del self.cur_instr

        # verify
        llvm_utils.verify_module(self.lmod)

        # optimize
        passes = ['mem2reg', 'simplifycfg', 'mergereturn']
        llvm_passes.run_function_passses(self.lfunc, passes)

    def incref(self, value):
        if not isinstance(value, llvm.Constant) or not value.isNullValue():
            self.call_abstract('incref', value)

    def decref(self, value):
        if not isinstance(value, llvm.Constant) or not value.isNullValue():
            self.call_abstract('decref', value)

    def op_LOAD_GLOBAL(self, namei):
        name = self.code.co_names[namei]
        value = self.get_global(name)
        self.incref(value)
        self.stack.append(value)

    def op_LOAD_FAST(self, var_num):
        name = self.code.co_varnames[var_num]
        value = self.builder.CreateLoad(self.symtab[name])
        self.incref(value)
        self.stack.append(value)

    def op_BINARY_ADD(self):
        tos = self.stack.pop()
        tos1 = self.stack.pop()
        value = self.call_abstract('binop.add', tos1, tos)
        self.decref(tos1)
        self.decref(tos)
        self.stack.append(value)

    def op_BINARY_SUBTRACT(self):
        tos = self.stack.pop()
        tos1 = self.stack.pop()
        value = self.call_abstract('binop.sub', tos1, tos)
        self.decref(tos1)
        self.decref(tos)
        self.stack.append(value)

    def op_BINARY_MULTIPLY(self):
        tos = self.stack.pop()
        tos1 = self.stack.pop()
        value = self.call_abstract('binop.mul', tos1, tos)
        self.decref(tos1)
        self.decref(tos)
        self.stack.append(value)

    def op_BINARY_DIVIDE(self):
        tos = self.stack.pop()
        tos1 = self.stack.pop()
        value = self.call_abstract('binop.div', tos1, tos)
        self.decref(tos1)
        self.decref(tos)
        self.stack.append(value)
    
    def op_RETURN_VALUE(self):
        tos = self.stack.pop()
        for obj in self.symtab.values():
            self.decref(self.builder.CreateLoad(obj))
        self.builder.CreateRet(tos)

    def op_COMPARE_OP(self, opname):
        tos = self.stack.pop()
        tos1 = self.stack.pop()
        op = dis.cmp_op[opname]
        opmap =  {'<' : 'lt',
                  '>' : 'gt',
                  '<=': 'le',
                  '>=': 'ge',
                  '==': 'eq',
                  '!=': 'ne',}
        opname = opmap[op]
        value = self.call_abstract_pred('cmpop.%s' % opname, tos1, tos)
        self.decref(tos1)
        self.decref(tos)
        self.stack.append(value)

    def op_POP_TOP(self):
        self.decref(self.stack.pop())

    def op_POP_JUMP_IF_FALSE(self, target):
        tos = self.stack.pop()
        true_block = self.get_or_make_block(self.cur_instr.offset +
                    opcode_info_table[dis.opmap['POP_JUMP_IF_FALSE']].oplen + 1)
        false_block = self.blocks[target]
        self.builder.CreateCondBr(tos, true_block, false_block)

    def op_LOAD_CONST(self, consti):
        const = self.code.co_consts[consti]
        if isinstance(const, int):
            constint = llvm.ConstantInt.get(llvm_types.i32, const, True)
            value = self.call_abstract('const', constint)
        elif isinstance(const, float):
            fptype = llvm_types.f64
            constfp = llvm.ConstantFP.get(fptype, const)
            value = self.call_abstract('const', constfp)
        elif const is None:
            gv = self.get_or_insert_global(None)
            value = self.builder.CreateLoad(gv)
        else:
            raise NotImplementedError(const, type(const))
        self.stack.append(value)

    def op_STORE_FAST(self, var_num):
        tos = self.stack.pop()
        var = self.code.co_varnames[var_num]
        ptr = self.symtab[var]
        self.decref(self.builder.CreateLoad(ptr))
        self.builder.CreateStore(tos, ptr)

    def op_SETUP_LOOP(self, delta):
        self.scope.append(defaultdict(list))

    def op_CALL_FUNCTION(self, argc):
        kwct = (argc >> 8) & 0xff
        posct = argc & 0xff
        assert not kwct
        args = list(reversed([self.stack.pop() for _ in range(posct)]))
        callee = self.stack.pop()
        kwargs = llvm_const.null(llvm_types.unknown_ptr)
        packed = self.call_abstract('pack', *args)
        value = self.call_abstract('call', callee, packed, kwargs)
        for arg in args:
            self.decref(arg)
        self.decref(kwargs)
        self.decref(packed)
        self.decref(callee)
        self.stack.append(value)

    def op_GET_ITER(self):
        tos = self.stack.pop()
        value = self.call_abstract('get_iter', tos)
        self.decref(tos)
        self.stack.append(value)

    def op_FOR_ITER(self, delta):
        instrdelta = self.cur_instr.length
        target = self.cur_instr.offset + delta + instrdelta
        tos = self.stack.pop()
        value = self.call_abstract('iter.next', tos)
        self.scope[-1]['decref'].append(tos)
        empty = self.call_abstract_pred('iter.empty', value)
        false_br = self.make_subblock(self.cur_instr.offset)
        true_br = self.blocks[target]
        self.builder.CreateCondBr(empty, true_br, false_br)
        self.builder.SetInsertPoint(false_br)
        self.stack.append(value)
    
    def op_INPLACE_ADD(self):
        tos = self.stack.pop()
        tos1 = self.stack.pop()
        value = self.call_abstract('binop.add', tos, tos1)
        self.decref(tos1)
        self.decref(tos)
        self.stack.append(value)

    def op_JUMP_ABSOLUTE(self, target):
        self.builder.CreateBr(self.blocks[target])

    def op_POP_BLOCK(self):
        scope = self.scope.pop()
        for obj in scope['decref']:
            self.decref(obj)

    def op_LOAD_ATTR(self, namei):
        tos = self.stack.pop()
        attr = self.code.co_names[namei]
        string = llvm.ConstantDataArray.getString(llvm_context, attr, True)
        name = '.str_co_names_%d' % namei
        linkage = llvm.GlobalVariable.LinkageTypes.InternalLinkage
        gv = self.lmod.getOrInsertGlobal(name, string.getType())
        gv = gv._downcast(llvm.GlobalVariable)
        if gv.getInitializer() is None:
            gv.setInitializer(string)
            gv.setLinkage(linkage)
            gv.setConstant(True)
        ptr = self.builder.CreateBitCast(gv, llvm_types.string)
        value = self.call_abstract('load_attr', tos, ptr)
        self.decref(tos)
        self.stack.append(value)

    def op_BINARY_SUBSCR(self):
        tos = self.stack.pop()
        tos1 = self.stack.pop()
        value = self.call_abstract('binary_subscr', tos1, tos)
        self.decref(tos1)
        self.decref(tos)
        self.stack.append(value)

    def op_BUILD_TUPLE(self, count):
        args = list(reversed([self.stack.pop() for i in range(count)]))
        value = self.call_abstract('build_tuple', *args)
        for arg in args:
            self.decref(arg)
        self.stack.append(value)

    def op_STORE_SUBSCR(self):
        tos = self.stack.pop()
        tos1 = self.stack.pop()
        tos2 = self.stack.pop()
        value = self.call_abstract('store_subscr', tos1, tos, tos2)
        self.decref(tos1)
        self.decref(tos)
        self.stack.append(value)

    def op_JUMP_FORWARD(self, offset):
        target = self.blocks[offset + self.cur_instr.offset +
                             self.cur_instr.length]
        self.builder.CreateBr(target)

    def op_DUP_TOPX(self, count):
        tmp = deque()
        for i in range(count):
            tmp.append(self.stack[count - i - 1])
        for t in reversed(tmp):
            self.incref(t)
            self.stack.append(t)

    def op_ROT_THREE(self):
        one = self.stack.pop()
        two = self.stack.pop()
        three = self.stack.pop()
        self.stack.append(one)
        self.stack.append(three)
        self.stack.append(two)

    def call_abstract(self, name, *args):
        is_vararg = False
        argtys = [x.getType() for x in args]
        retty = llvm_types.unknown_ptr
        mangledname = mangle('numba.abstract.%s' % name, argtys)
        callee = llvm_utils.get_or_insert_func(self.lmod, mangledname,
                                               retty, argtys)
        return self.builder.CreateCall(callee, args)

    def call_abstract_pred(self, name, *args):
        is_vararg = False
        argtys = [x.getType() for x in args]
        retty = llvm_types.i1
        fnty = llvm.FunctionType.get(retty, argtys, is_vararg)
        mangledname = mangle('numba.abstract.%s' % name, argtys)
        callee = llvm_utils.get_or_insert_func(self.lmod, mangledname,
                                               retty, argtys)
        return self.builder.CreateCall(callee, args)

    def make_block(self, offset=None):
        name = 'block%d' % offset if offset is not None else ''
        return llvm_utils.make_basic_block(self.lfunc, name)

    def make_subblock(self, offset=None):
        name = 'subblock%d' % offset if offset is not None else ''
        return llvm_utils.make_basic_block(self.lfunc, name)

    def get_or_make_block(self, offset):
        if offset not in self.blocks:
            self.blocks[offset] = self.make_block(offset)
        return self.blocks[offset]

    def get_global(self, name):
        if name in self.globals:
            gname = 'global.' + name
        else:
            gname = builtins_table[self.builtins[name]]
        gv = self.get_or_insert_global(gname)
        return self.builder.CreateLoad(gv)


    def get_or_insert_global(self, gname):
        gv = self.lmod.getOrInsertGlobal('numba.%s' % gname,
                                         llvm_types.unknown_ptr)
        return gv

def translate(func):
    trans = Translator(func)
    trans.execute()
    return trans

def remove_redundant_refct(lmod):
    for lfunc in lmod.list_functions():
        if not lfunc.isDeclaration():
            for bb in lfunc.getBasicBlockList():
                increflist = defaultdict(deque)
                decreflist = defaultdict(deque)
                # find increfs and decrefs
                for ip, inst in enumerate(bb.getInstList()):
                    if inst.getOpcodeName() == 'call':
                        callinst = inst._downcast(llvm.CallInst)
                        calledfname = callinst.getCalledFunction().getName()
                        if calledfname.startswith('numba.abstract.incref.'):
                            sym = callinst.getArgOperand(0)
                            increflist[sym].append((ip, callinst))
                        if calledfname.startswith('numba.abstract.decref.'):
                            sym = callinst.getArgOperand(0)
                            decreflist[sym].append((ip, callinst))
                # mark redundant inst
                eraselist = []
                for k, vl in increflist.items():
                    if k in decreflist:
                        for incref_ip, incref in vl:
                            if decreflist[k]:
                                decref_ip, decref = decreflist[k].popleft()
                                if incref_ip < decref_ip:
                                    eraselist.append(incref)
                                    eraselist.append(decref)
                # delete all marked inst
                for inst in eraselist:
                    inst.eraseFromParent()

