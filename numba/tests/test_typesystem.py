from __future__ import print_function, absolute_import
from numba import unittest_support as unittest
from numba.typesystem import typesystem


class TestTypeSystem(unittest.TestCase):
    def test_type_create(self):
        context = typesystem.TypeContext()
        int32 = context.get_type('int32')
        float32 = context.get_type('float32')
        self.assertEqual(str(int32), "int32")
        self.assertEqual(float32.name, "float32")

    def test_cast(self):
        context = typesystem.TypeContext()
        int32 = context.get_type('int32')
        float32 = context.get_type('float32')
        int32_to_float32 = context.cast(int32, float32)
        self.assertTrue(int32_to_float32.is_coerce)
        self.assertFalse(int32_to_float32.is_exact)
        self.assertFalse(int32_to_float32.is_promote)
        self.assertTrue(int32_to_float32.distance > 0)

    def test_coerce(self):
        context = typesystem.TypeContext()
        int32 = context.get_type('int32')
        float32 = context.get_type('float32')
        float64 = context.get_type('float64')
        coerced, safe = context.coerce(int32, float32, float64)
        self.assertEqual(coerced, float64)
        self.assertTrue(safe)

    def test_overload(self):
        context = typesystem.TypeContext()

        int32 = context.get_type('int32')
        float32 = context.get_type('float32')

        sig = int32, float32
        ver0 = int32, int32
        ver1 = float32, float32
        vers = ver0, ver1
        self.assertEqual(context.resolve(sig, vers), vers)
        self.assertEqual(context.best_resolve(sig, vers), ver0)

if __name__ == '__main__':
    unittest.main()
