import support
from usecases import *

class TestUseCase1(support.TestCase):
    def test1(self):
        self.run_template(usecase1, args=(1, 2, 3))

    def test2(self):
        self.run_template(usecase1, args=(1., 2., 3.))

    def test3(self):
        self.run_template(usecase1, args=(32, 432., 3j))

class TestUseCase2(support.TestCase):
    def test1(self):
        self.run_template(usecase2, args=(1, 2, 3))

    def test2(self):
        self.run_template(usecase2, args=(1., 2., 3.))

class TestUseCase3(support.TestCase):
    def test1(self):
        self.run_template(usecase3, args=(1, 2))

    def test2(self):
        self.run_template(usecase3, args=(1, 2.))

class TestUseCase4(support.TestCase):
    def test1(self):
        self.run_template(usecase4, args=(1, 2, 3))

    def test2(self):
        self.run_template(usecase4, args=(1, 2, 3.))

class TestUseCase5(support.TestCase):
    def test1(self):
        self.run_template(usecase5, args=(1, 2, 3))

    def test2(self):
        self.run_template(usecase5, args=(1., 2., 3.))

class TestUseCase6(support.TestCase):
    def test1(self):
        self.run_template(usecase5, args=(1, 2, 3))

    def test2(self):
        self.run_template(usecase5, args=(1., 2., 3.))

class TestUseCase7(support.TestCase):
    def test1(self):
        self.run_template(usecase5, args=(1, 2, 3))

    def test2(self):
        self.run_template(usecase5, args=(1., 2., 3.))

if __name__ == '__main__':
    support.main()
