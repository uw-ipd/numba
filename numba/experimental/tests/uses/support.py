import os, unittest, numpy as np
from ... import pipeline, execution

class TestCase(unittest.TestCase):
    def run_template(self, func, args=(), allclose=False):
        debug_flag = int(os.environ.get('DEBUG', 0))
        ci = pipeline.compile(func, debug=debug_flag)
        exe = execution.ExecutionContext(ci.lmod)
        callable = exe.prepare(ci.lfunc, globals=ci.func.func_globals)
        got = callable(*args)
        exp = func(*args)
        if allclose:
            self.assertEqual(got, exp)
        else:
            self.assertTrue(np.allclose(got, exp))

def main():
    unittest.main()
