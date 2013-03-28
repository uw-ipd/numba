builtins_table = {
    range:  'builtin.range',
    xrange: 'builtin.xrange',
    None:   'builtin.none',
}

inv_builtins_table = dict((v, k) for k, v in builtins_table.items())
