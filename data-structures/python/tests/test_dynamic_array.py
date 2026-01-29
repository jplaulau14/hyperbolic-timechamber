import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynamic_array import DynamicArray


class TestDynamicArray(unittest.TestCase):
    def test_default_construction(self):
        arr = DynamicArray()
        self.assertEqual(arr.size(), 0)
        self.assertEqual(arr.capacity(), 0)
        self.assertTrue(arr.empty())

    def test_sized_construction(self):
        arr = DynamicArray(5)
        self.assertEqual(arr.size(), 5)
        self.assertEqual(arr.capacity(), 5)
        self.assertFalse(arr.empty())
        for i in range(arr.size()):
            self.assertIsNone(arr[i])

    def test_push_back_single(self):
        arr = DynamicArray()
        arr.push_back(42)
        self.assertEqual(arr.size(), 1)
        self.assertEqual(arr[0], 42)

    def test_push_back_multiple(self):
        arr = DynamicArray()
        arr.push_back(1)
        arr.push_back(2)
        arr.push_back(3)
        self.assertEqual(arr.size(), 3)
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[1], 2)
        self.assertEqual(arr[2], 3)

    def test_push_back_triggers_growth(self):
        arr = DynamicArray()
        arr.push_back(1)
        self.assertEqual(arr.capacity(), 1)
        arr.push_back(2)
        self.assertEqual(arr.capacity(), 2)
        arr.push_back(3)
        self.assertEqual(arr.capacity(), 4)
        arr.push_back(4)
        arr.push_back(5)
        self.assertEqual(arr.capacity(), 8)
        self.assertEqual(arr.size(), 5)

    def test_pop_back(self):
        arr = DynamicArray()
        arr.push_back(10)
        arr.push_back(20)
        arr.push_back(30)
        arr.pop_back()
        self.assertEqual(arr.size(), 2)
        self.assertEqual(arr.back(), 20)
        arr.pop_back()
        self.assertEqual(arr.size(), 1)
        self.assertEqual(arr.back(), 10)

    def test_at_valid_index(self):
        arr = DynamicArray()
        arr.push_back(100)
        arr.push_back(200)
        arr.push_back(300)
        self.assertEqual(arr.at(0), 100)
        self.assertEqual(arr.at(1), 200)
        self.assertEqual(arr.at(2), 300)
        arr.set_at(1, 999)
        self.assertEqual(arr.at(1), 999)

    def test_at_throws_out_of_range(self):
        arr = DynamicArray()
        arr.push_back(1)
        with self.assertRaises(IndexError):
            arr.at(1)
        with self.assertRaises(IndexError):
            arr.at(100)
        empty = DynamicArray()
        with self.assertRaises(IndexError):
            empty.at(0)

    def test_subscript_operator(self):
        arr = DynamicArray()
        arr.push_back(10)
        arr.push_back(20)
        self.assertEqual(arr[0], 10)
        self.assertEqual(arr[1], 20)
        arr[0] = 99
        self.assertEqual(arr[0], 99)

    def test_front_and_back(self):
        arr = DynamicArray()
        arr.push_back(1)
        arr.push_back(2)
        arr.push_back(3)
        self.assertEqual(arr.front(), 1)
        self.assertEqual(arr.back(), 3)

    def test_reserve_increases_capacity(self):
        arr = DynamicArray()
        arr.reserve(10)
        self.assertGreaterEqual(arr.capacity(), 10)
        self.assertEqual(arr.size(), 0)

    def test_reserve_preserves_elements(self):
        arr = DynamicArray()
        arr.push_back(1)
        arr.push_back(2)
        arr.push_back(3)
        arr.reserve(100)
        self.assertGreaterEqual(arr.capacity(), 100)
        self.assertEqual(arr.size(), 3)
        self.assertEqual(arr[0], 1)
        self.assertEqual(arr[1], 2)
        self.assertEqual(arr[2], 3)

    def test_reserve_smaller_is_noop(self):
        arr = DynamicArray()
        arr.reserve(10)
        cap = arr.capacity()
        arr.reserve(5)
        self.assertEqual(arr.capacity(), cap)

    def test_clear_resets_size_not_capacity(self):
        arr = DynamicArray()
        arr.push_back(1)
        arr.push_back(2)
        arr.push_back(3)
        cap = arr.capacity()
        arr.clear()
        self.assertEqual(arr.size(), 0)
        self.assertTrue(arr.empty())
        self.assertEqual(arr.capacity(), cap)

    def test_copy(self):
        arr = DynamicArray()
        arr.push_back(1)
        arr.push_back(2)
        arr.push_back(3)

        clone = arr.copy()
        self.assertEqual(clone.size(), 3)
        self.assertEqual(clone[0], 1)
        self.assertEqual(clone[1], 2)
        self.assertEqual(clone[2], 3)

        arr[0] = 999
        self.assertEqual(clone[0], 1)

    def test_iteration(self):
        arr = DynamicArray()
        arr.push_back(1)
        arr.push_back(2)
        arr.push_back(3)
        arr.push_back(4)
        total = sum(arr)
        self.assertEqual(total, 10)

    def test_non_trivial_type(self):
        arr = DynamicArray()
        arr.push_back("hello")
        arr.push_back("world")
        self.assertEqual(arr.size(), 2)
        self.assertEqual(arr[0], "hello")
        self.assertEqual(arr[1], "world")
        self.assertEqual(arr.front(), "hello")
        self.assertEqual(arr.back(), "world")

    def test_data_reference(self):
        arr = DynamicArray()
        arr.push_back(1)
        arr.push_back(2)
        d = arr.data()
        self.assertEqual(d[0], 1)
        self.assertEqual(d[1], 2)

    def test_empty_data(self):
        arr = DynamicArray()
        self.assertIsNone(arr.data())

    def test_len(self):
        arr = DynamicArray()
        self.assertEqual(len(arr), 0)
        arr.push_back(1)
        arr.push_back(2)
        self.assertEqual(len(arr), 2)

    def test_negative_index_at(self):
        arr = DynamicArray()
        arr.push_back(1)
        with self.assertRaises(IndexError):
            arr.at(-1)

    def test_set_at_out_of_range(self):
        arr = DynamicArray()
        arr.push_back(1)
        with self.assertRaises(IndexError):
            arr.set_at(1, 0)
        with self.assertRaises(IndexError):
            arr.set_at(-1, 0)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            DynamicArray(-1)
        with self.assertRaises(ValueError):
            DynamicArray("three")

    def test_front_empty(self):
        arr = DynamicArray()
        with self.assertRaises(IndexError):
            arr.front()

    def test_back_empty(self):
        arr = DynamicArray()
        with self.assertRaises(IndexError):
            arr.back()

    def test_pop_back_empty(self):
        arr = DynamicArray()
        with self.assertRaises(IndexError):
            arr.pop_back()


if __name__ == "__main__":
    unittest.main()
