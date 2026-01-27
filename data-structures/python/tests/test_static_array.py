import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from static_array import StaticArray


class TestStaticArray(unittest.TestCase):
    def test_size_and_empty(self):
        arr = StaticArray(5)
        self.assertEqual(arr.size(), 5)
        self.assertFalse(arr.empty())

        empty_arr = StaticArray(0)
        self.assertEqual(empty_arr.size(), 0)
        self.assertTrue(empty_arr.empty())

    def test_fill_and_access(self):
        arr = StaticArray(4)
        arr.fill(42)
        for i in range(arr.size()):
            self.assertEqual(arr[i], 42)

    def test_subscript_operator(self):
        arr = StaticArray(3)
        arr[0] = 10
        arr[1] = 20
        arr[2] = 30
        self.assertEqual(arr[0], 10)
        self.assertEqual(arr[1], 20)
        self.assertEqual(arr[2], 30)

    def test_at_valid_index(self):
        arr = StaticArray(3)
        arr.set_at(0, 100)
        arr.set_at(1, 200)
        arr.set_at(2, 300)
        self.assertEqual(arr.at(0), 100)
        self.assertEqual(arr.at(1), 200)
        self.assertEqual(arr.at(2), 300)

    def test_at_throws_out_of_range(self):
        arr = StaticArray(3)
        with self.assertRaises(IndexError):
            arr.at(3)
        with self.assertRaises(IndexError):
            arr.at(100)

    def test_at_throws_on_zero_size(self):
        arr = StaticArray(0)
        with self.assertRaises(IndexError):
            arr.at(0)

    def test_front_and_back(self):
        arr = StaticArray(4, 0)
        arr[0] = 1
        arr[3] = 99
        self.assertEqual(arr.front(), 1)
        self.assertEqual(arr.back(), 99)

    def test_data_reference(self):
        arr = StaticArray(3)
        arr.fill(7)
        d = arr.data()
        self.assertEqual(d[0], 7)
        self.assertEqual(d[1], 7)
        self.assertEqual(d[2], 7)

        d[1] = 42
        self.assertEqual(arr[1], 42)

    def test_data_zero_size(self):
        arr = StaticArray(0)
        self.assertIsNone(arr.data())

    def test_range_based_for(self):
        arr = StaticArray(5)
        arr.fill(3)
        total = sum(arr)
        self.assertEqual(total, 15)

    def test_iteration(self):
        arr = StaticArray(4)
        for i in range(4):
            arr[i] = i * 10

        it = iter(arr)
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 10)

    def test_non_trivial_type(self):
        arr = StaticArray(3)
        arr[0] = "hello"
        arr[1] = "world"
        arr[2] = "!"
        self.assertEqual(arr[0], "hello")
        self.assertEqual(arr.at(1), "world")
        self.assertEqual(arr.back(), "!")

    def test_fill_overwrites(self):
        arr = StaticArray(3)
        arr.fill(1)
        arr.fill(2)
        for i in range(arr.size()):
            self.assertEqual(arr[i], 2)

    def test_zero_size_iteration(self):
        arr = StaticArray(0)
        count = 0
        for _ in arr:
            count += 1
        self.assertEqual(count, 0)

    def test_single_element(self):
        arr = StaticArray(1)
        arr[0] = 42
        self.assertEqual(arr.front(), 42)
        self.assertEqual(arr.back(), 42)
        self.assertEqual(arr.size(), 1)

    def test_len(self):
        arr = StaticArray(5)
        self.assertEqual(len(arr), 5)

        empty = StaticArray(0)
        self.assertEqual(len(empty), 0)

    def test_negative_index_at(self):
        arr = StaticArray(3)
        arr.fill(0)
        with self.assertRaises(IndexError):
            arr.at(-1)

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            StaticArray(-1)
        with self.assertRaises(ValueError):
            StaticArray("three")


if __name__ == "__main__":
    unittest.main()
