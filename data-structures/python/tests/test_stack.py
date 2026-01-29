import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stack import Stack


class TestStack(unittest.TestCase):
    def test_new_stack_is_empty(self):
        s = Stack()
        self.assertEqual(s.size(), 0)
        self.assertTrue(s.is_empty())

    def test_top_on_empty_stack_raises(self):
        s = Stack()
        with self.assertRaises(IndexError):
            s.top()

    def test_pop_on_empty_stack_raises(self):
        s = Stack()
        with self.assertRaises(IndexError):
            s.pop()

    def test_push_single_element(self):
        s = Stack()
        s.push(42)
        self.assertEqual(s.size(), 1)

    def test_push_multiple_elements(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        self.assertEqual(s.size(), 3)

    def test_push_many_elements(self):
        s = Stack()
        for i in range(100):
            s.push(i)
        self.assertEqual(s.size(), 100)

    def test_pop_returns_top_element(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        self.assertEqual(s.pop(), 3)

    def test_pop_decrements_size(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.pop()
        self.assertEqual(s.size(), 1)

    def test_pop_all_elements_until_empty(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        s.pop()
        s.pop()
        s.pop()
        self.assertTrue(s.is_empty())

    def test_lifo_order(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        self.assertEqual(s.pop(), 3)
        self.assertEqual(s.pop(), 2)
        self.assertEqual(s.pop(), 1)

    def test_top_returns_top_without_removing(self):
        s = Stack()
        s.push(10)
        s.push(20)
        self.assertEqual(s.top(), 20)
        self.assertEqual(s.size(), 2)

    def test_top_after_push_shows_new_element(self):
        s = Stack()
        s.push(1)
        self.assertEqual(s.top(), 1)
        s.push(2)
        self.assertEqual(s.top(), 2)

    def test_top_multiple_times_returns_same_value(self):
        s = Stack()
        s.push(42)
        self.assertEqual(s.top(), 42)
        self.assertEqual(s.top(), 42)
        self.assertEqual(s.top(), 42)
        self.assertEqual(s.size(), 1)

    def test_clear_makes_stack_empty(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        s.clear()
        self.assertTrue(s.is_empty())
        self.assertEqual(s.size(), 0)

    def test_clear_on_empty_stack_is_noop(self):
        s = Stack()
        s.clear()
        self.assertTrue(s.is_empty())

    def test_can_push_after_clear(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.clear()
        s.push(99)
        self.assertEqual(s.size(), 1)
        self.assertEqual(s.top(), 99)

    def test_size_returns_correct_count_after_pushes(self):
        s = Stack()
        self.assertEqual(s.size(), 0)
        s.push(1)
        self.assertEqual(s.size(), 1)
        s.push(2)
        self.assertEqual(s.size(), 2)
        s.push(3)
        self.assertEqual(s.size(), 3)

    def test_size_returns_correct_count_after_pops(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        self.assertEqual(s.size(), 3)
        s.pop()
        self.assertEqual(s.size(), 2)
        s.pop()
        self.assertEqual(s.size(), 1)

    def test_is_empty_true_only_when_size_is_zero(self):
        s = Stack()
        self.assertTrue(s.is_empty())
        s.push(1)
        self.assertFalse(s.is_empty())
        s.pop()
        self.assertTrue(s.is_empty())

    def test_copy_creates_independent_copy(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        clone = s.copy()
        self.assertEqual(clone.size(), 3)
        self.assertEqual(clone.top(), 3)

    def test_push_to_original_does_not_affect_clone(self):
        s = Stack()
        s.push(1)
        s.push(2)
        clone = s.copy()
        s.push(3)
        self.assertEqual(s.size(), 3)
        self.assertEqual(clone.size(), 2)

    def test_pop_from_original_does_not_affect_clone(self):
        s = Stack()
        s.push(1)
        s.push(2)
        s.push(3)
        clone = s.copy()
        s.pop()
        self.assertEqual(s.size(), 2)
        self.assertEqual(clone.size(), 3)
        self.assertEqual(clone.top(), 3)

    def test_works_with_strings(self):
        s = Stack()
        s.push("hello")
        s.push("world")
        self.assertEqual(s.top(), "world")
        self.assertEqual(s.pop(), "world")
        self.assertEqual(s.pop(), "hello")

    def test_works_with_non_primitive_types(self):
        s = Stack()
        s.push([1, 2, 3])
        s.push({"key": "value"})
        self.assertEqual(s.top(), {"key": "value"})
        self.assertEqual(s.pop(), {"key": "value"})
        self.assertEqual(s.pop(), [1, 2, 3])

    def test_single_element_push_then_pop(self):
        s = Stack()
        s.push(42)
        self.assertEqual(s.pop(), 42)
        self.assertTrue(s.is_empty())

    def test_alternating_push_pop_operations(self):
        s = Stack()
        s.push(1)
        self.assertEqual(s.pop(), 1)
        s.push(2)
        s.push(3)
        self.assertEqual(s.pop(), 3)
        s.push(4)
        self.assertEqual(s.pop(), 4)
        self.assertEqual(s.pop(), 2)
        self.assertTrue(s.is_empty())

    def test_large_number_of_elements(self):
        s = Stack()
        for i in range(1000):
            s.push(i)
        self.assertEqual(s.size(), 1000)
        for i in range(999, -1, -1):
            self.assertEqual(s.pop(), i)
        self.assertTrue(s.is_empty())

    def test_len_dunder(self):
        s = Stack()
        self.assertEqual(len(s), 0)
        s.push(1)
        s.push(2)
        self.assertEqual(len(s), 2)

    def test_bool_dunder(self):
        s = Stack()
        self.assertFalse(bool(s))
        s.push(1)
        self.assertTrue(bool(s))
        s.pop()
        self.assertFalse(bool(s))


if __name__ == "__main__":
    unittest.main()
