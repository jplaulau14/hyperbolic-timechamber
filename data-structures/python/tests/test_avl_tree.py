import unittest
import math
from src.avl_tree import AVLTree


class TestAVLTreeConstruction(unittest.TestCase):
    def test_new_tree_is_empty(self):
        tree: AVLTree[int] = AVLTree()
        self.assertEqual(tree.size(), 0)
        self.assertTrue(tree.is_empty())
        self.assertEqual(tree.height(), 0)

    def test_min_on_empty_tree_raises(self):
        tree: AVLTree[int] = AVLTree()
        with self.assertRaises(ValueError):
            tree.min()

    def test_max_on_empty_tree_raises(self):
        tree: AVLTree[int] = AVLTree()
        with self.assertRaises(ValueError):
            tree.max()


class TestAVLTreeInsert(unittest.TestCase):
    def test_insert_single_element(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        self.assertEqual(tree.size(), 1)
        self.assertEqual(tree.height(), 1)
        self.assertTrue(tree.contains(10))

    def test_insert_multiple_elements(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        self.assertEqual(tree.size(), 3)
        self.assertTrue(tree.contains(10))
        self.assertTrue(tree.contains(5))
        self.assertTrue(tree.contains(15))

    def test_insert_duplicate_is_no_op(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(10)
        self.assertEqual(tree.size(), 1)

    def test_insert_maintains_bst_property(self):
        tree: AVLTree[int] = AVLTree()
        values = [50, 30, 70, 20, 40, 60, 80]
        for v in values:
            tree.insert(v)
        self.assertEqual(tree.in_order(), sorted(values))

    def test_insert_maintains_balance(self):
        tree: AVLTree[int] = AVLTree()
        for i in range(100):
            tree.insert(i)
        self.assertTrue(tree.is_balanced())


class TestAVLTreeRightRotation(unittest.TestCase):
    def test_ll_imbalance_triggers_right_rotation(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(30)
        tree.insert(20)
        tree.insert(10)
        self.assertTrue(tree.is_balanced())
        self.assertEqual(tree.in_order(), [10, 20, 30])

    def test_height_after_right_rotation(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(30)
        tree.insert(20)
        tree.insert(10)
        self.assertEqual(tree.height(), 2)


class TestAVLTreeLeftRotation(unittest.TestCase):
    def test_rr_imbalance_triggers_left_rotation(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(20)
        tree.insert(30)
        self.assertTrue(tree.is_balanced())
        self.assertEqual(tree.in_order(), [10, 20, 30])

    def test_height_after_left_rotation(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(20)
        tree.insert(30)
        self.assertEqual(tree.height(), 2)


class TestAVLTreeLeftRightRotation(unittest.TestCase):
    def test_lr_imbalance_triggers_left_right_rotation(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(30)
        tree.insert(10)
        tree.insert(20)
        self.assertTrue(tree.is_balanced())
        self.assertEqual(tree.in_order(), [10, 20, 30])

    def test_height_after_left_right_rotation(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(30)
        tree.insert(10)
        tree.insert(20)
        self.assertEqual(tree.height(), 2)


class TestAVLTreeRightLeftRotation(unittest.TestCase):
    def test_rl_imbalance_triggers_right_left_rotation(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(30)
        tree.insert(20)
        self.assertTrue(tree.is_balanced())
        self.assertEqual(tree.in_order(), [10, 20, 30])

    def test_height_after_right_left_rotation(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(30)
        tree.insert(20)
        self.assertEqual(tree.height(), 2)


class TestAVLTreeContains(unittest.TestCase):
    def test_contains_returns_true_for_existing(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        self.assertTrue(tree.contains(10))
        self.assertTrue(tree.contains(5))
        self.assertTrue(tree.contains(15))

    def test_contains_returns_false_for_nonexistent(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        self.assertFalse(tree.contains(5))

    def test_contains_after_insert(self):
        tree: AVLTree[int] = AVLTree()
        self.assertFalse(tree.contains(10))
        tree.insert(10)
        self.assertTrue(tree.contains(10))

    def test_contains_after_remove(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        self.assertTrue(tree.contains(10))
        tree.remove(10)
        self.assertFalse(tree.contains(10))


class TestAVLTreeRemove(unittest.TestCase):
    def test_remove_leaf_node(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        tree.remove(5)
        self.assertEqual(tree.size(), 2)
        self.assertFalse(tree.contains(5))
        self.assertTrue(tree.is_balanced())

    def test_remove_node_with_left_child(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        tree.insert(3)
        tree.remove(5)
        self.assertFalse(tree.contains(5))
        self.assertTrue(tree.contains(3))
        self.assertTrue(tree.is_balanced())

    def test_remove_node_with_right_child(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        tree.insert(7)
        tree.remove(5)
        self.assertFalse(tree.contains(5))
        self.assertTrue(tree.contains(7))
        self.assertTrue(tree.is_balanced())

    def test_remove_node_with_two_children(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        tree.insert(3)
        tree.insert(7)
        tree.remove(5)
        self.assertFalse(tree.contains(5))
        self.assertTrue(tree.contains(3))
        self.assertTrue(tree.contains(7))
        self.assertTrue(tree.is_balanced())

    def test_remove_root_node(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        tree.remove(10)
        self.assertFalse(tree.contains(10))
        self.assertEqual(tree.size(), 2)
        self.assertTrue(tree.is_balanced())

    def test_remove_nonexistent_is_no_op(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.remove(5)
        self.assertEqual(tree.size(), 1)

    def test_size_decrements_after_remove(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        self.assertEqual(tree.size(), 2)
        tree.remove(5)
        self.assertEqual(tree.size(), 1)

    def test_remove_triggers_rebalancing(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(50)
        tree.insert(30)
        tree.insert(70)
        tree.insert(20)
        tree.insert(40)
        tree.insert(60)
        tree.insert(80)
        tree.insert(10)
        tree.remove(60)
        tree.remove(70)
        tree.remove(80)
        self.assertTrue(tree.is_balanced())


class TestAVLTreeMinMax(unittest.TestCase):
    def test_min_returns_smallest(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(50)
        tree.insert(30)
        tree.insert(70)
        tree.insert(20)
        self.assertEqual(tree.min(), 20)

    def test_max_returns_largest(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(50)
        tree.insert(30)
        tree.insert(70)
        tree.insert(80)
        self.assertEqual(tree.max(), 80)

    def test_min_max_after_insert(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(50)
        self.assertEqual(tree.min(), 50)
        self.assertEqual(tree.max(), 50)
        tree.insert(10)
        self.assertEqual(tree.min(), 10)
        tree.insert(90)
        self.assertEqual(tree.max(), 90)

    def test_min_max_after_remove(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(50)
        tree.insert(30)
        tree.insert(70)
        tree.remove(30)
        self.assertEqual(tree.min(), 50)
        tree.remove(70)
        self.assertEqual(tree.max(), 50)


class TestAVLTreeHeight(unittest.TestCase):
    def test_height_of_empty_tree(self):
        tree: AVLTree[int] = AVLTree()
        self.assertEqual(tree.height(), 0)

    def test_height_of_single_node(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        self.assertEqual(tree.height(), 1)

    def test_height_updates_after_insert(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        self.assertEqual(tree.height(), 1)
        tree.insert(5)
        self.assertEqual(tree.height(), 2)
        tree.insert(15)
        self.assertEqual(tree.height(), 2)

    def test_height_updates_after_remove(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        tree.remove(5)
        tree.remove(15)
        self.assertEqual(tree.height(), 1)

    def test_height_is_logarithmic_for_sorted_insert(self):
        tree: AVLTree[int] = AVLTree()
        n = 1000
        for i in range(n):
            tree.insert(i)
        max_height = 1.44 * math.log2(n + 2)
        self.assertLessEqual(tree.height(), max_height)


class TestAVLTreeTraversals(unittest.TestCase):
    def test_in_order_yields_sorted(self):
        tree: AVLTree[int] = AVLTree()
        values = [50, 30, 70, 20, 40, 60, 80]
        for v in values:
            tree.insert(v)
        self.assertEqual(tree.in_order(), sorted(values))

    def test_pre_order_yields_correct_sequence(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(20)
        tree.insert(10)
        tree.insert(30)
        result = tree.pre_order()
        self.assertEqual(result[0], 20)
        self.assertIn(10, result)
        self.assertIn(30, result)

    def test_post_order_yields_correct_sequence(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(20)
        tree.insert(10)
        tree.insert(30)
        result = tree.post_order()
        self.assertEqual(result[-1], 20)
        self.assertIn(10, result)
        self.assertIn(30, result)

    def test_traversals_on_empty_tree(self):
        tree: AVLTree[int] = AVLTree()
        self.assertEqual(tree.in_order(), [])
        self.assertEqual(tree.pre_order(), [])
        self.assertEqual(tree.post_order(), [])


class TestAVLTreeClear(unittest.TestCase):
    def test_clear_makes_tree_empty(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        tree.clear()
        self.assertEqual(tree.size(), 0)
        self.assertEqual(tree.height(), 0)
        self.assertTrue(tree.is_empty())

    def test_clear_on_empty_is_no_op(self):
        tree: AVLTree[int] = AVLTree()
        tree.clear()
        self.assertEqual(tree.size(), 0)
        self.assertEqual(tree.height(), 0)


class TestAVLTreeAVLProperty(unittest.TestCase):
    def test_bst_property_holds_after_operations(self):
        tree: AVLTree[int] = AVLTree()
        for i in [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45]:
            tree.insert(i)
        tree.remove(30)
        tree.remove(70)
        in_order = tree.in_order()
        self.assertEqual(in_order, sorted(in_order))

    def test_balance_factor_valid_after_operations(self):
        tree: AVLTree[int] = AVLTree()
        for i in range(50):
            tree.insert(i)
        for i in range(0, 50, 2):
            tree.remove(i)
        self.assertTrue(tree.is_balanced())

    def test_sorted_insert_produces_balanced_tree(self):
        tree: AVLTree[int] = AVLTree()
        for i in range(100):
            tree.insert(i)
        self.assertTrue(tree.is_balanced())


class TestAVLTreeCopy(unittest.TestCase):
    def test_copy_creates_independent_copy(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        tree.insert(15)
        clone = tree.copy()
        self.assertEqual(clone.in_order(), tree.in_order())
        self.assertEqual(clone.size(), tree.size())

    def test_insert_to_original_does_not_affect_clone(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        clone = tree.copy()
        tree.insert(5)
        self.assertFalse(clone.contains(5))
        self.assertEqual(clone.size(), 1)

    def test_remove_from_original_does_not_affect_clone(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        tree.insert(5)
        clone = tree.copy()
        tree.remove(5)
        self.assertTrue(clone.contains(5))
        self.assertEqual(clone.size(), 2)


class TestAVLTreeEdgeCases(unittest.TestCase):
    def test_single_element_tree(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(42)
        self.assertEqual(tree.size(), 1)
        self.assertEqual(tree.height(), 1)
        self.assertEqual(tree.min(), 42)
        self.assertEqual(tree.max(), 42)
        self.assertTrue(tree.is_balanced())

    def test_sorted_insert_produces_balanced(self):
        tree: AVLTree[int] = AVLTree()
        for i in range(1, 101):
            tree.insert(i)
        self.assertTrue(tree.is_balanced())
        self.assertEqual(tree.in_order(), list(range(1, 101)))

    def test_reverse_sorted_insert_produces_balanced(self):
        tree: AVLTree[int] = AVLTree()
        for i in range(100, 0, -1):
            tree.insert(i)
        self.assertTrue(tree.is_balanced())
        self.assertEqual(tree.in_order(), list(range(1, 101)))

    def test_large_number_of_elements(self):
        tree: AVLTree[int] = AVLTree()
        n = 1000
        for i in range(n):
            tree.insert(i)
        self.assertEqual(tree.size(), n)
        max_height = 1.44 * math.log2(n + 2)
        self.assertLessEqual(tree.height(), max_height)
        self.assertTrue(tree.is_balanced())

    def test_negative_numbers(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(-10)
        tree.insert(-20)
        tree.insert(-5)
        self.assertEqual(tree.min(), -20)
        self.assertEqual(tree.max(), -5)
        self.assertTrue(tree.is_balanced())

    def test_remove_all_elements_one_by_one(self):
        tree: AVLTree[int] = AVLTree()
        values = [50, 30, 70, 20, 40, 60, 80]
        for v in values:
            tree.insert(v)
        for v in values:
            tree.remove(v)
            self.assertTrue(tree.is_balanced())
        self.assertTrue(tree.is_empty())

    def test_alternating_insert_remove_maintains_balance(self):
        tree: AVLTree[int] = AVLTree()
        for i in range(100):
            tree.insert(i)
            if i % 3 == 0 and i > 0:
                tree.remove(i - 1)
            self.assertTrue(tree.is_balanced())


class TestAVLTreeDunderMethods(unittest.TestCase):
    def test_len(self):
        tree: AVLTree[int] = AVLTree()
        self.assertEqual(len(tree), 0)
        tree.insert(10)
        self.assertEqual(len(tree), 1)
        tree.insert(5)
        self.assertEqual(len(tree), 2)

    def test_contains(self):
        tree: AVLTree[int] = AVLTree()
        tree.insert(10)
        self.assertIn(10, tree)
        self.assertNotIn(5, tree)

    def test_iter(self):
        tree: AVLTree[int] = AVLTree()
        values = [30, 10, 50, 20, 40]
        for v in values:
            tree.insert(v)
        self.assertEqual(list(tree), sorted(values))


if __name__ == '__main__':
    unittest.main()
