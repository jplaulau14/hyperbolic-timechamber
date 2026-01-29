import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from linked_list import LinkedList


class TestLinkedList(unittest.TestCase):
    def test_new_list_is_empty(self):
        lst = LinkedList()
        self.assertEqual(lst.size(), 0)
        self.assertTrue(lst.is_empty())
        self.assertEqual(len(lst), 0)

    def test_front_on_empty_raises(self):
        lst = LinkedList()
        with self.assertRaises(IndexError):
            lst.front()

    def test_back_on_empty_raises(self):
        lst = LinkedList()
        with self.assertRaises(IndexError):
            lst.back()

    def test_push_front_single(self):
        lst = LinkedList()
        lst.push_front(42)
        self.assertEqual(lst.size(), 1)
        self.assertEqual(lst.front(), 42)
        self.assertEqual(lst.back(), 42)

    def test_push_front_multiple(self):
        lst = LinkedList()
        lst.push_front(1)
        lst.push_front(2)
        lst.push_front(3)
        self.assertEqual(lst.size(), 3)
        self.assertEqual(lst.front(), 3)
        self.assertEqual(lst.back(), 1)
        self.assertEqual(list(lst), [3, 2, 1])

    def test_push_back_single(self):
        lst = LinkedList()
        lst.push_back(42)
        self.assertEqual(lst.size(), 1)
        self.assertEqual(lst.front(), 42)
        self.assertEqual(lst.back(), 42)

    def test_push_back_multiple(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        self.assertEqual(lst.size(), 3)
        self.assertEqual(lst.front(), 1)
        self.assertEqual(lst.back(), 3)
        self.assertEqual(list(lst), [1, 2, 3])

    def test_mixed_push_front_and_back(self):
        lst = LinkedList()
        lst.push_back(2)
        lst.push_front(1)
        lst.push_back(3)
        lst.push_front(0)
        self.assertEqual(list(lst), [0, 1, 2, 3])

    def test_pop_front_returns_correct_value(self):
        lst = LinkedList()
        lst.push_back(10)
        lst.push_back(20)
        lst.push_back(30)
        self.assertEqual(lst.pop_front(), 10)
        self.assertEqual(lst.size(), 2)
        self.assertEqual(lst.front(), 20)

    def test_pop_front_until_empty(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        self.assertEqual(lst.pop_front(), 1)
        self.assertEqual(lst.pop_front(), 2)
        self.assertEqual(lst.pop_front(), 3)
        self.assertTrue(lst.is_empty())

    def test_pop_front_on_empty_raises(self):
        lst = LinkedList()
        with self.assertRaises(IndexError):
            lst.pop_front()

    def test_pop_back_returns_correct_value(self):
        lst = LinkedList()
        lst.push_back(10)
        lst.push_back(20)
        lst.push_back(30)
        self.assertEqual(lst.pop_back(), 30)
        self.assertEqual(lst.size(), 2)
        self.assertEqual(lst.back(), 20)

    def test_pop_back_until_empty(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        self.assertEqual(lst.pop_back(), 3)
        self.assertEqual(lst.pop_back(), 2)
        self.assertEqual(lst.pop_back(), 1)
        self.assertTrue(lst.is_empty())

    def test_pop_back_on_empty_raises(self):
        lst = LinkedList()
        with self.assertRaises(IndexError):
            lst.pop_back()

    def test_front_does_not_remove(self):
        lst = LinkedList()
        lst.push_back(42)
        self.assertEqual(lst.front(), 42)
        self.assertEqual(lst.size(), 1)

    def test_back_does_not_remove(self):
        lst = LinkedList()
        lst.push_back(42)
        self.assertEqual(lst.back(), 42)
        self.assertEqual(lst.size(), 1)

    def test_at_first_element(self):
        lst = LinkedList()
        lst.push_back(10)
        lst.push_back(20)
        lst.push_back(30)
        self.assertEqual(lst.at(0), 10)

    def test_at_last_element(self):
        lst = LinkedList()
        lst.push_back(10)
        lst.push_back(20)
        lst.push_back(30)
        self.assertEqual(lst.at(2), 30)

    def test_at_middle_element(self):
        lst = LinkedList()
        lst.push_back(10)
        lst.push_back(20)
        lst.push_back(30)
        self.assertEqual(lst.at(1), 20)

    def test_at_negative_raises(self):
        lst = LinkedList()
        lst.push_back(1)
        with self.assertRaises(IndexError):
            lst.at(-1)

    def test_at_out_of_range_raises(self):
        lst = LinkedList()
        lst.push_back(1)
        with self.assertRaises(IndexError):
            lst.at(1)
        with self.assertRaises(IndexError):
            lst.at(100)

    def test_at_invalid_type_raises(self):
        lst = LinkedList()
        lst.push_back(1)
        with self.assertRaises(TypeError):
            lst.at("zero")

    def test_insert_at_zero(self):
        lst = LinkedList()
        lst.push_back(2)
        lst.push_back(3)
        lst.insert_at(0, 1)
        self.assertEqual(list(lst), [1, 2, 3])
        self.assertEqual(lst.front(), 1)

    def test_insert_at_size(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.insert_at(2, 3)
        self.assertEqual(list(lst), [1, 2, 3])
        self.assertEqual(lst.back(), 3)

    def test_insert_at_middle(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(3)
        lst.insert_at(1, 2)
        self.assertEqual(list(lst), [1, 2, 3])

    def test_insert_at_invalid_raises(self):
        lst = LinkedList()
        lst.push_back(1)
        with self.assertRaises(IndexError):
            lst.insert_at(-1, 0)
        with self.assertRaises(IndexError):
            lst.insert_at(2, 0)

    def test_insert_at_invalid_type_raises(self):
        lst = LinkedList()
        with self.assertRaises(TypeError):
            lst.insert_at("zero", 1)

    def test_remove_at_zero(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        self.assertEqual(lst.remove_at(0), 1)
        self.assertEqual(list(lst), [2, 3])
        self.assertEqual(lst.front(), 2)

    def test_remove_at_last(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        self.assertEqual(lst.remove_at(2), 3)
        self.assertEqual(list(lst), [1, 2])
        self.assertEqual(lst.back(), 2)

    def test_remove_at_middle(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        self.assertEqual(lst.remove_at(1), 2)
        self.assertEqual(list(lst), [1, 3])

    def test_remove_at_invalid_raises(self):
        lst = LinkedList()
        lst.push_back(1)
        with self.assertRaises(IndexError):
            lst.remove_at(-1)
        with self.assertRaises(IndexError):
            lst.remove_at(1)

    def test_remove_at_invalid_type_raises(self):
        lst = LinkedList()
        lst.push_back(1)
        with self.assertRaises(TypeError):
            lst.remove_at("zero")

    def test_clear(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        lst.clear()
        self.assertTrue(lst.is_empty())
        self.assertEqual(lst.size(), 0)

    def test_clear_on_empty(self):
        lst = LinkedList()
        lst.clear()
        self.assertTrue(lst.is_empty())

    def test_iteration_order(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        result = []
        for value in lst:
            result.append(value)
        self.assertEqual(result, [1, 2, 3])

    def test_iteration_empty(self):
        lst = LinkedList()
        result = list(lst)
        self.assertEqual(result, [])

    def test_copy_creates_independent_copy(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        lst.push_back(3)
        clone = lst.copy()
        self.assertEqual(list(clone), [1, 2, 3])
        self.assertEqual(clone.size(), 3)

    def test_modifying_original_does_not_affect_clone(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        clone = lst.copy()
        lst.push_back(3)
        lst.pop_front()
        self.assertEqual(list(clone), [1, 2])
        self.assertEqual(list(lst), [2, 3])

    def test_works_with_strings(self):
        lst = LinkedList()
        lst.push_back("hello")
        lst.push_back("world")
        self.assertEqual(lst.size(), 2)
        self.assertEqual(lst.front(), "hello")
        self.assertEqual(lst.back(), "world")
        self.assertEqual(list(lst), ["hello", "world"])

    def test_single_element_list(self):
        lst = LinkedList()
        lst.push_back(42)
        self.assertEqual(lst.front(), lst.back())
        self.assertEqual(lst.pop_front(), 42)
        self.assertTrue(lst.is_empty())

    def test_two_element_list(self):
        lst = LinkedList()
        lst.push_back(1)
        lst.push_back(2)
        self.assertEqual(lst.front(), 1)
        self.assertEqual(lst.back(), 2)
        self.assertEqual(lst.pop_back(), 2)
        self.assertEqual(lst.front(), lst.back())
        self.assertEqual(lst.size(), 1)


if __name__ == "__main__":
    unittest.main()
