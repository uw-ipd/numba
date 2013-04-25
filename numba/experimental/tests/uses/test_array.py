import numpy as np
import support
from usecases import *

class TestUseCase8(support.TestCase):
    def test1(self):
        A = np.arange(10)
        B = np.arange(10, 20)
        C = np.arange(20, 30)
        self.run_template(usecase8, args=(A, B, C))

class TestUseCase9(support.TestCase):
    def test1(self):
        A = np.arange(10)
        B = np.arange(10, 20)
        C = np.arange(20, 30)
        self.run_template(usecase9, args=(A, B, C))


if __name__ == '__main__':
    support.main()
