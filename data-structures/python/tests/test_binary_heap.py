import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from binary_heap import BinaryHeap


class TestBinaryHeap(unittest.TestCase):

    # Construction & Basic State
    def test_new_heap_is_empty(self):
        heap = BinaryHeap()
        self.assertEqual(heap.size(), 0)
        self.assertTrue(heap.is_empty())

    def test_peek_on_empty_heap_raises(self):
        heap = BinaryHeap()
        with self.assertRaises(IndexError):
            heap.peek()

    def test_pop_on_empty_heap_raises(self):
        heap = BinaryHeap()
        with self.assertRaises(IndexError):
            heap.pop()

    # Push Operations
    def test_push_single_element(self):
        heap = BinaryHeap()
        heap.push(42)
        self.assertEqual(heap.size(), 1)
        self.assertEqual(heap.peek(), 42)

    def test_push_multiple_elements_maintains_heap_property(self):
        heap = BinaryHeap()
        heap.push(30)
        heap.push(10)
        heap.push(20)
        self.assertEqual(heap.peek(), 10)

    def test_push_elements_in_ascending_order(self):
        heap = BinaryHeap()
        for i in range(1, 11):
            heap.push(i)
        self.assertEqual(heap.peek(), 1)

    def test_push_elements_in_descending_order(self):
        heap = BinaryHeap()
        for i in range(10, 0, -1):
            heap.push(i)
        self.assertEqual(heap.peek(), 1)

    def test_push_elements_in_random_order(self):
        heap = BinaryHeap()
        values = [5, 3, 8, 1, 9, 2, 7, 4, 6]
        for v in values:
            heap.push(v)
        self.assertEqual(heap.peek(), 1)

    # Pop Operations
    def test_pop_returns_minimum_element(self):
        heap = BinaryHeap()
        heap.push(30)
        heap.push(10)
        heap.push(20)
        self.assertEqual(heap.pop(), 10)

    def test_pop_removes_element_and_restores_heap_property(self):
        heap = BinaryHeap()
        heap.push(30)
        heap.push(10)
        heap.push(20)
        heap.pop()
        self.assertEqual(heap.peek(), 20)

    def test_pop_all_elements_yields_sorted_order(self):
        heap = BinaryHeap()
        values = [5, 3, 8, 1, 9, 2, 7, 4, 6]
        for v in values:
            heap.push(v)
        result = []
        while not heap.is_empty():
            result.append(heap.pop())
        self.assertEqual(result, sorted(values))

    def test_size_decrements_after_pop(self):
        heap = BinaryHeap()
        heap.push(1)
        heap.push(2)
        heap.push(3)
        self.assertEqual(heap.size(), 3)
        heap.pop()
        self.assertEqual(heap.size(), 2)

    # Peek Operations
    def test_peek_returns_minimum_without_removing(self):
        heap = BinaryHeap()
        heap.push(30)
        heap.push(10)
        heap.push(20)
        self.assertEqual(heap.peek(), 10)
        self.assertEqual(heap.size(), 3)

    def test_multiple_peeks_return_same_value(self):
        heap = BinaryHeap()
        heap.push(42)
        self.assertEqual(heap.peek(), 42)
        self.assertEqual(heap.peek(), 42)
        self.assertEqual(heap.peek(), 42)

    def test_peek_after_push_shows_new_min_if_applicable(self):
        heap = BinaryHeap()
        heap.push(10)
        self.assertEqual(heap.peek(), 10)
        heap.push(5)
        self.assertEqual(heap.peek(), 5)
        heap.push(15)
        self.assertEqual(heap.peek(), 5)

    # Heapify / From Array
    def test_from_array_builds_heap_from_unsorted_array(self):
        heap = BinaryHeap.from_array([5, 3, 8, 1, 9, 2])
        self.assertEqual(heap.peek(), 1)

    def test_from_array_has_correct_min_at_top(self):
        heap = BinaryHeap.from_array([100, 50, 75, 25, 10])
        self.assertEqual(heap.peek(), 10)

    def test_from_array_all_elements_present(self):
        values = [5, 3, 8, 1, 9, 2]
        heap = BinaryHeap.from_array(values)
        result = []
        while not heap.is_empty():
            result.append(heap.pop())
        self.assertEqual(sorted(result), sorted(values))

    # Clear
    def test_clear_makes_heap_empty(self):
        heap = BinaryHeap()
        heap.push(1)
        heap.push(2)
        heap.push(3)
        heap.clear()
        self.assertTrue(heap.is_empty())
        self.assertEqual(heap.size(), 0)

    def test_clear_on_empty_heap_is_noop(self):
        heap = BinaryHeap()
        heap.clear()
        self.assertTrue(heap.is_empty())

    # Heap Property Verification
    def test_heap_property_after_operations(self):
        heap = BinaryHeap()
        values = [5, 3, 8, 1, 9, 2, 7, 4, 6]
        for v in values:
            heap.push(v)
            self._verify_heap_property(heap)
        while not heap.is_empty():
            heap.pop()
            self._verify_heap_property(heap)

    def test_heap_property_with_duplicates(self):
        heap = BinaryHeap()
        values = [5, 3, 5, 3, 1, 1, 5]
        for v in values:
            heap.push(v)
        self._verify_heap_property(heap)
        self.assertEqual(heap.peek(), 1)

    def test_heap_property_with_negative_numbers(self):
        heap = BinaryHeap()
        values = [5, -3, 8, -1, 9, -2]
        for v in values:
            heap.push(v)
        self._verify_heap_property(heap)
        self.assertEqual(heap.peek(), -3)

    def _verify_heap_property(self, heap):
        data = heap._data
        for i in range(len(data)):
            left = 2 * i + 1
            right = 2 * i + 2
            if left < len(data):
                self.assertLessEqual(data[i], data[left])
            if right < len(data):
                self.assertLessEqual(data[i], data[right])

    # Copy/Clone
    def test_copy_creates_independent_copy(self):
        heap = BinaryHeap()
        heap.push(30)
        heap.push(10)
        heap.push(20)
        clone = heap.copy()
        self.assertEqual(clone.size(), 3)
        self.assertEqual(clone.peek(), 10)

    def test_push_to_original_does_not_affect_clone(self):
        heap = BinaryHeap()
        heap.push(30)
        heap.push(10)
        clone = heap.copy()
        heap.push(5)
        self.assertEqual(heap.peek(), 5)
        self.assertEqual(clone.peek(), 10)
        self.assertEqual(clone.size(), 2)

    def test_pop_from_original_does_not_affect_clone(self):
        heap = BinaryHeap()
        heap.push(30)
        heap.push(10)
        heap.push(20)
        clone = heap.copy()
        heap.pop()
        self.assertEqual(heap.peek(), 20)
        self.assertEqual(clone.peek(), 10)
        self.assertEqual(clone.size(), 3)

    # Non-trivial Types
    def test_works_with_floating_point_numbers(self):
        heap = BinaryHeap()
        heap.push(3.14)
        heap.push(1.41)
        heap.push(2.72)
        self.assertEqual(heap.peek(), 1.41)
        self.assertEqual(heap.pop(), 1.41)
        self.assertEqual(heap.pop(), 2.72)
        self.assertEqual(heap.pop(), 3.14)

    def test_works_with_strings(self):
        heap = BinaryHeap()
        heap.push("banana")
        heap.push("apple")
        heap.push("cherry")
        self.assertEqual(heap.pop(), "apple")
        self.assertEqual(heap.pop(), "banana")
        self.assertEqual(heap.pop(), "cherry")

    # Edge Cases
    def test_single_element_heap(self):
        heap = BinaryHeap()
        heap.push(42)
        self.assertEqual(heap.peek(), 42)
        self.assertEqual(heap.pop(), 42)
        self.assertTrue(heap.is_empty())

    def test_two_element_heap(self):
        heap = BinaryHeap()
        heap.push(20)
        heap.push(10)
        self.assertEqual(heap.pop(), 10)
        self.assertEqual(heap.pop(), 20)

    def test_large_number_of_elements(self):
        heap = BinaryHeap()
        for i in range(1000, 0, -1):
            heap.push(i)
        self.assertEqual(heap.size(), 1000)
        self.assertEqual(heap.peek(), 1)
        result = []
        while not heap.is_empty():
            result.append(heap.pop())
        self.assertEqual(result, list(range(1, 1001)))

    def test_many_push_pop_cycles(self):
        heap = BinaryHeap()
        for _ in range(100):
            heap.push(5)
            heap.push(3)
            heap.push(8)
            self.assertEqual(heap.pop(), 3)
            heap.push(1)
            self.assertEqual(heap.pop(), 1)
            heap.pop()
            heap.pop()
        self.assertTrue(heap.is_empty())

    # Dunder Methods
    def test_len_dunder(self):
        heap = BinaryHeap()
        self.assertEqual(len(heap), 0)
        heap.push(1)
        heap.push(2)
        self.assertEqual(len(heap), 2)

    def test_bool_dunder(self):
        heap = BinaryHeap()
        self.assertFalse(bool(heap))
        heap.push(1)
        self.assertTrue(bool(heap))
        heap.pop()
        self.assertFalse(bool(heap))

    # From Array Edge Cases
    def test_from_array_empty_array(self):
        heap = BinaryHeap.from_array([])
        self.assertTrue(heap.is_empty())

    def test_from_array_single_element(self):
        heap = BinaryHeap.from_array([42])
        self.assertEqual(heap.peek(), 42)
        self.assertEqual(heap.size(), 1)


if __name__ == "__main__":
    unittest.main()
