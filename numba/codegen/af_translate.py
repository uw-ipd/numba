#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ______________________________________________________________________
# Module imports

from __future__ import print_function, division, absolute_import

import inspect

import llvm.core as lc

from llpython.bytecode_visitor import GenericFlowVisitor

# ______________________________________________________________________
# Class definition(s)

class AddressFlowTranslator(GenericFlowVisitor):

    def translate_cfg(self, code_obj, cfg, llvm_module):
        assert inspect.iscode(code_obj)
        self.code_obj = code_obj
        self.cfg = cfg
        if llvm_module is None:
            llvm_module = lc.Module.new('_af_translate_%d' % id(self))
        self.llvm_module = llvm_module
        self.visit(cfg.blocks)
        del self.llvm_module
        del self.cfg
        del self.code_obj
        return llvm_module

    def enter_block(self, block):
        return True

# ______________________________________________________________________
# End of af_translate.py
