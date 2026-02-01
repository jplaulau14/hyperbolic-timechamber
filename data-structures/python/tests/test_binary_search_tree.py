import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from binary_search_tree import BinarySearchTree


class TestBinarySearchTree(unittest.TestCase):

    def test_new_tree_is_empty(self):
        bst = BinarySearchTree()
        self.assertEqual(bst.size(), 0)
        self.assertTrue(bst.is_empty())

    def test_min_on_empty_raises(self):
        bst = BinarySearchTree()
        with self.assertRaises(ValueError):
            bst.min()

    def test_max_on_empty_raises(self):
        bst = BinarySearchTree()
        with self.assertRaises(ValueError):
            bst.max()

    def test_insert_single_element(self):
        bst = BinarySearchTree()
        bst.insert(42)
        self.assertEqual(bst.size(), 1)
        self.assertFalse(bst.is_empty())
        self.assertTrue(bst.contains(42))

    def test_insert_multiple_elements(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        self.assertEqual(bst.size(), 3)
        self.assertTrue(bst.contains(50))
        self.assertTrue(bst.contains(30))
        self.assertTrue(bst.contains(70))

    def test_insert_duplicate_is_noop(self):
        bst = BinarySearchTree()
        bst.insert(42)
        bst.insert(42)
        self.assertEqual(bst.size(), 1)

    def test_insert_maintains_bst_property(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(20)
        bst.insert(40)
        result = bst.in_order()
        self.assertEqual(result, sorted(result))

    def test_contains_returns_true_for_existing(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        self.assertTrue(bst.contains(50))
        self.assertTrue(bst.contains(30))
        self.assertTrue(bst.contains(70))

    def test_contains_returns_false_for_nonexistent(self):
        bst = BinarySearchTree()
        bst.insert(50)
        self.assertFalse(bst.contains(30))
        self.assertFalse(bst.contains(70))

    def test_contains_after_insert(self):
        bst = BinarySearchTree()
        self.assertFalse(bst.contains(42))
        bst.insert(42)
        self.assertTrue(bst.contains(42))

    def test_contains_after_remove(self):
        bst = BinarySearchTree()
        bst.insert(42)
        self.assertTrue(bst.contains(42))
        bst.remove(42)
        self.assertFalse(bst.contains(42))

    def test_remove_leaf_node(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.remove(30)
        self.assertFalse(bst.contains(30))
        self.assertTrue(bst.contains(50))
        self.assertTrue(bst.contains(70))
        self.assertEqual(bst.size(), 2)

    def test_remove_node_with_left_child(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(20)
        bst.remove(30)
        self.assertFalse(bst.contains(30))
        self.assertTrue(bst.contains(50))
        self.assertTrue(bst.contains(20))

    def test_remove_node_with_right_child(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(40)
        bst.remove(30)
        self.assertFalse(bst.contains(30))
        self.assertTrue(bst.contains(50))
        self.assertTrue(bst.contains(40))

    def test_remove_node_with_two_children(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(20)
        bst.insert(40)
        bst.remove(30)
        self.assertFalse(bst.contains(30))
        self.assertTrue(bst.contains(50))
        self.assertTrue(bst.contains(70))
        self.assertTrue(bst.contains(20))
        self.assertTrue(bst.contains(40))

    def test_remove_root_node(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.remove(50)
        self.assertFalse(bst.contains(50))
        self.assertTrue(bst.contains(30))
        self.assertTrue(bst.contains(70))

    def test_remove_nonexistent_is_noop(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.remove(100)
        self.assertEqual(bst.size(), 1)
        self.assertTrue(bst.contains(50))

    def test_size_decrements_after_remove(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        self.assertEqual(bst.size(), 3)
        bst.remove(30)
        self.assertEqual(bst.size(), 2)

    def test_min_returns_smallest(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(20)
        self.assertEqual(bst.min(), 20)

    def test_max_returns_largest(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(80)
        self.assertEqual(bst.max(), 80)

    def test_min_max_after_insert(self):
        bst = BinarySearchTree()
        bst.insert(50)
        self.assertEqual(bst.min(), 50)
        self.assertEqual(bst.max(), 50)
        bst.insert(30)
        self.assertEqual(bst.min(), 30)
        bst.insert(70)
        self.assertEqual(bst.max(), 70)

    def test_min_max_after_remove(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.remove(30)
        self.assertEqual(bst.min(), 50)
        bst.remove(70)
        self.assertEqual(bst.max(), 50)

    def test_in_order_yields_sorted(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(20)
        bst.insert(40)
        self.assertEqual(bst.in_order(), [20, 30, 40, 50, 70])

    def test_pre_order_yields_correct_sequence(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(20)
        bst.insert(40)
        self.assertEqual(bst.pre_order(), [50, 30, 20, 40, 70])

    def test_post_order_yields_correct_sequence(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(20)
        bst.insert(40)
        self.assertEqual(bst.post_order(), [20, 40, 30, 70, 50])

    def test_traversals_on_empty_tree(self):
        bst = BinarySearchTree()
        self.assertEqual(bst.in_order(), [])
        self.assertEqual(bst.pre_order(), [])
        self.assertEqual(bst.post_order(), [])

    def test_clear_makes_tree_empty(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.clear()
        self.assertTrue(bst.is_empty())
        self.assertEqual(bst.size(), 0)

    def test_clear_on_empty_is_noop(self):
        bst = BinarySearchTree()
        bst.clear()
        self.assertTrue(bst.is_empty())

    def test_bst_property_holds_after_operations(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(20)
        bst.insert(40)
        result = bst.in_order()
        self.assertEqual(result, sorted(result))
        bst.remove(30)
        result = bst.in_order()
        self.assertEqual(result, sorted(result))

    def test_copy_creates_independent_copy(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        clone = bst.copy()
        self.assertEqual(clone.size(), 3)
        self.assertTrue(clone.contains(50))
        self.assertTrue(clone.contains(30))
        self.assertTrue(clone.contains(70))

    def test_insert_to_original_doesnt_affect_copy(self):
        bst = BinarySearchTree()
        bst.insert(50)
        clone = bst.copy()
        bst.insert(30)
        self.assertEqual(bst.size(), 2)
        self.assertEqual(clone.size(), 1)
        self.assertFalse(clone.contains(30))

    def test_remove_from_original_doesnt_affect_copy(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        clone = bst.copy()
        bst.remove(30)
        self.assertFalse(bst.contains(30))
        self.assertTrue(clone.contains(30))

    def test_single_element_tree(self):
        bst = BinarySearchTree()
        bst.insert(42)
        self.assertEqual(bst.min(), 42)
        self.assertEqual(bst.max(), 42)
        self.assertTrue(bst.contains(42))
        bst.remove(42)
        self.assertTrue(bst.is_empty())

    def test_sorted_insert_degenerates_to_list(self):
        bst = BinarySearchTree()
        for i in range(1, 11):
            bst.insert(i)
        self.assertEqual(bst.size(), 10)
        self.assertEqual(bst.min(), 1)
        self.assertEqual(bst.max(), 10)

    def test_large_number_of_elements(self):
        bst = BinarySearchTree()
        for i in range(1, 1001):
            bst.insert(i)
        self.assertEqual(bst.size(), 1000)
        self.assertEqual(bst.min(), 1)
        self.assertEqual(bst.max(), 1000)

    def test_negative_numbers(self):
        bst = BinarySearchTree()
        bst.insert(-10)
        bst.insert(0)
        bst.insert(10)
        bst.insert(-20)
        self.assertEqual(bst.min(), -20)
        self.assertEqual(bst.max(), 10)
        self.assertTrue(bst.contains(-10))

    def test_remove_all_elements_one_by_one(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        bst.insert(20)
        bst.insert(40)
        bst.remove(20)
        bst.remove(40)
        bst.remove(30)
        bst.remove(70)
        bst.remove(50)
        self.assertTrue(bst.is_empty())

    def test_len_dunder(self):
        bst = BinarySearchTree()
        self.assertEqual(len(bst), 0)
        bst.insert(50)
        bst.insert(30)
        self.assertEqual(len(bst), 2)

    def test_contains_dunder(self):
        bst = BinarySearchTree()
        bst.insert(50)
        self.assertTrue(50 in bst)
        self.assertFalse(30 in bst)

    def test_iter_dunder(self):
        bst = BinarySearchTree()
        bst.insert(50)
        bst.insert(30)
        bst.insert(70)
        result = list(bst)
        self.assertEqual(result, [30, 50, 70])

    def test_works_with_strings(self):
        bst = BinarySearchTree()
        bst.insert("banana")
        bst.insert("apple")
        bst.insert("cherry")
        self.assertEqual(bst.min(), "apple")
        self.assertEqual(bst.max(), "cherry")
        self.assertEqual(bst.in_order(), ["apple", "banana", "cherry"])

    def test_works_with_floats(self):
        bst = BinarySearchTree()
        bst.insert(3.14)
        bst.insert(1.41)
        bst.insert(2.71)
        self.assertEqual(bst.min(), 1.41)
        self.assertEqual(bst.max(), 3.14)


if __name__ == "__main__":
    unittest.main()
