import dis
from collections import namedtuple

OpcodeInfo = namedtuple('OpcodeInfo', ['opname', 'oplen'])


def _make_opcode_info_table(seq):
    def gen():
        for opname, numoperands in seq:
            yield dis.opmap[opname], OpcodeInfo(opname, numoperands)
    return dict(gen())

opcode_info_table = _make_opcode_info_table([
# opname, operandlen
('BINARY_ADD', 0),
('BINARY_DIVIDE', 0),
('BINARY_MULTIPLY', 0),
('BINARY_SUBSCR', 0),
('BINARY_SUBTRACT', 0),
('BUILD_TUPLE', 2),
('CALL_FUNCTION', 2),
('COMPARE_OP', 2),
('DUP_TOPX', 2),
('FOR_ITER', 2),
('GET_ITER', 0),
('INPLACE_ADD', 0),
('JUMP_ABSOLUTE', 2),
('JUMP_FORWARD', 2),
('LOAD_ATTR', 2),
('LOAD_CONST', 2),
('LOAD_FAST', 2),
('LOAD_GLOBAL', 2),
('POP_BLOCK', 0),
('POP_JUMP_IF_FALSE', 2),
('POP_TOP', 0),
('RETURN_VALUE', 0),
('ROT_THREE', 0),
('SETUP_LOOP', 2),
('STORE_FAST', 2),
('STORE_SUBSCR', 0),
])
