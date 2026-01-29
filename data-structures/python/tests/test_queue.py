import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from queue import Queue


class TestQueue(unittest.TestCase):
    def test_new_queue_is_empty(self):
        q = Queue()
        self.assertEqual(q.size(), 0)
        self.assertTrue(q.is_empty())

    def test_front_on_empty_raises(self):
        q = Queue()
        with self.assertRaises(IndexError):
            q.front()

    def test_back_on_empty_raises(self):
        q = Queue()
        with self.assertRaises(IndexError):
            q.back()

    def test_dequeue_on_empty_raises(self):
        q = Queue()
        with self.assertRaises(IndexError):
            q.dequeue()

    def test_enqueue_single_element(self):
        q = Queue()
        q.enqueue(42)
        self.assertEqual(q.size(), 1)
        self.assertFalse(q.is_empty())

    def test_enqueue_multiple_elements(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        self.assertEqual(q.size(), 3)

    def test_front_and_back_after_multiple_enqueues(self):
        q = Queue()
        q.enqueue(10)
        q.enqueue(20)
        q.enqueue(30)
        self.assertEqual(q.front(), 10)
        self.assertEqual(q.back(), 30)

    def test_dequeue_returns_front_element(self):
        q = Queue()
        q.enqueue(100)
        q.enqueue(200)
        self.assertEqual(q.dequeue(), 100)

    def test_dequeue_decrements_size(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        self.assertEqual(q.size(), 2)
        q.dequeue()
        self.assertEqual(q.size(), 1)

    def test_dequeue_all_elements_until_empty(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        q.dequeue()
        q.dequeue()
        q.dequeue()
        self.assertTrue(q.is_empty())

    def test_fifo_order(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        self.assertEqual(q.dequeue(), 1)
        self.assertEqual(q.dequeue(), 2)
        self.assertEqual(q.dequeue(), 3)

    def test_front_does_not_remove(self):
        q = Queue()
        q.enqueue(42)
        self.assertEqual(q.front(), 42)
        self.assertEqual(q.size(), 1)
        self.assertEqual(q.front(), 42)

    def test_back_does_not_remove(self):
        q = Queue()
        q.enqueue(42)
        self.assertEqual(q.back(), 42)
        self.assertEqual(q.size(), 1)
        self.assertEqual(q.back(), 42)

    def test_front_and_back_same_with_single_element(self):
        q = Queue()
        q.enqueue(99)
        self.assertEqual(q.front(), 99)
        self.assertEqual(q.back(), 99)

    def test_circular_buffer_wrap_around(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        q.enqueue(4)
        q.dequeue()
        q.dequeue()
        q.enqueue(5)
        q.enqueue(6)
        self.assertEqual(q.front(), 3)
        self.assertEqual(q.back(), 6)
        self.assertEqual(q.size(), 4)

    def test_fill_dequeue_some_enqueue_more(self):
        q = Queue()
        for i in range(4):
            q.enqueue(i)
        q.dequeue()
        q.dequeue()
        q.enqueue(10)
        q.enqueue(11)
        self.assertEqual(q.dequeue(), 2)
        self.assertEqual(q.dequeue(), 3)
        self.assertEqual(q.dequeue(), 10)
        self.assertEqual(q.dequeue(), 11)

    def test_growth_preserves_order(self):
        q = Queue()
        for i in range(10):
            q.enqueue(i)
        for i in range(10):
            self.assertEqual(q.dequeue(), i)

    def test_clear_makes_queue_empty(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        q.clear()
        self.assertTrue(q.is_empty())
        self.assertEqual(q.size(), 0)

    def test_clear_on_empty_queue(self):
        q = Queue()
        q.clear()
        self.assertTrue(q.is_empty())

    def test_enqueue_after_clear(self):
        q = Queue()
        q.enqueue(1)
        q.clear()
        q.enqueue(2)
        self.assertEqual(q.front(), 2)
        self.assertEqual(q.size(), 1)

    def test_size_after_enqueues(self):
        q = Queue()
        for i in range(5):
            q.enqueue(i)
            self.assertEqual(q.size(), i + 1)

    def test_size_after_dequeues(self):
        q = Queue()
        for i in range(5):
            q.enqueue(i)
        for i in range(5):
            self.assertEqual(q.size(), 5 - i)
            q.dequeue()

    def test_is_empty_only_when_size_zero(self):
        q = Queue()
        self.assertTrue(q.is_empty())
        q.enqueue(1)
        self.assertFalse(q.is_empty())
        q.dequeue()
        self.assertTrue(q.is_empty())

    def test_copy_creates_independent_copy(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        q.enqueue(3)
        clone = q.copy()
        self.assertEqual(clone.size(), 3)
        self.assertEqual(clone.front(), 1)
        self.assertEqual(clone.back(), 3)

    def test_enqueue_to_original_does_not_affect_clone(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        clone = q.copy()
        q.enqueue(3)
        self.assertEqual(q.size(), 3)
        self.assertEqual(clone.size(), 2)

    def test_dequeue_from_original_does_not_affect_clone(self):
        q = Queue()
        q.enqueue(1)
        q.enqueue(2)
        clone = q.copy()
        q.dequeue()
        self.assertEqual(q.front(), 2)
        self.assertEqual(clone.front(), 1)

    def test_works_with_strings(self):
        q = Queue()
        q.enqueue("hello")
        q.enqueue("world")
        self.assertEqual(q.front(), "hello")
        self.assertEqual(q.back(), "world")
        self.assertEqual(q.dequeue(), "hello")

    def test_works_with_non_primitive_types(self):
        q = Queue()
        q.enqueue([1, 2, 3])
        q.enqueue({"key": "value"})
        self.assertEqual(q.front(), [1, 2, 3])
        self.assertEqual(q.back(), {"key": "value"})

    def test_single_element_enqueue_dequeue(self):
        q = Queue()
        q.enqueue(42)
        self.assertEqual(q.dequeue(), 42)
        self.assertTrue(q.is_empty())

    def test_alternating_enqueue_dequeue(self):
        q = Queue()
        for i in range(10):
            q.enqueue(i)
            self.assertEqual(q.dequeue(), i)
        self.assertTrue(q.is_empty())

    def test_large_number_of_elements(self):
        q = Queue()
        for i in range(1000):
            q.enqueue(i)
        self.assertEqual(q.size(), 1000)
        for i in range(1000):
            self.assertEqual(q.dequeue(), i)
        self.assertTrue(q.is_empty())

    def test_many_wrap_around_cycles(self):
        q = Queue()
        for cycle in range(100):
            for i in range(10):
                q.enqueue(cycle * 10 + i)
            for i in range(10):
                self.assertEqual(q.dequeue(), cycle * 10 + i)
        self.assertTrue(q.is_empty())

    def test_len_dunder(self):
        q = Queue()
        self.assertEqual(len(q), 0)
        q.enqueue(1)
        q.enqueue(2)
        self.assertEqual(len(q), 2)

    def test_bool_dunder(self):
        q = Queue()
        self.assertFalse(bool(q))
        q.enqueue(1)
        self.assertTrue(bool(q))
        q.dequeue()
        self.assertFalse(bool(q))


if __name__ == "__main__":
    unittest.main()
