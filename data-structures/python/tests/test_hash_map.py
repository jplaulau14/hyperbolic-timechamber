import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hash_map import HashMap


class TestHashMapConstruction(unittest.TestCase):
    def test_new_map_is_empty(self):
        m = HashMap()
        self.assertEqual(m.size(), 0)
        self.assertTrue(m.is_empty())

    def test_default_capacity(self):
        m = HashMap()
        self.assertEqual(m._capacity, 16)


class TestHashMapInsert(unittest.TestCase):
    def test_insert_single_pair(self):
        m = HashMap()
        m.put("key", "value")
        self.assertEqual(m.size(), 1)
        self.assertEqual(m.get("key"), "value")

    def test_insert_multiple_pairs(self):
        m = HashMap()
        m.put("a", 1)
        m.put("b", 2)
        m.put("c", 3)
        self.assertEqual(m.size(), 3)
        self.assertEqual(m.get("a"), 1)
        self.assertEqual(m.get("b"), 2)
        self.assertEqual(m.get("c"), 3)

    def test_insert_duplicate_updates_value(self):
        m = HashMap()
        m.put("key", "old")
        m.put("key", "new")
        self.assertEqual(m.size(), 1)
        self.assertEqual(m.get("key"), "new")

    def test_insert_triggers_rehash(self):
        m = HashMap(4)
        m.put("a", 1)
        m.put("b", 2)
        m.put("c", 3)
        old_capacity = m._capacity
        m.put("d", 4)
        self.assertGreater(m._capacity, old_capacity)


class TestHashMapGet(unittest.TestCase):
    def test_get_existing_key(self):
        m = HashMap()
        m.put("key", "value")
        self.assertEqual(m.get("key"), "value")

    def test_get_nonexistent_key_raises(self):
        m = HashMap()
        with self.assertRaises(KeyError):
            m.get("missing")

    def test_get_after_update(self):
        m = HashMap()
        m.put("key", "old")
        m.put("key", "new")
        self.assertEqual(m.get("key"), "new")

    def test_get_or_existing(self):
        m = HashMap()
        m.put("key", "value")
        self.assertEqual(m.get_or("key", "default"), "value")

    def test_get_or_missing(self):
        m = HashMap()
        self.assertEqual(m.get_or("missing", "default"), "default")


class TestHashMapRemove(unittest.TestCase):
    def test_remove_existing_key(self):
        m = HashMap()
        m.put("key", "value")
        result = m.remove("key")
        self.assertEqual(result, "value")
        self.assertFalse(m.contains("key"))

    def test_remove_nonexistent_key(self):
        m = HashMap()
        result = m.remove("missing")
        self.assertIsNone(result)

    def test_remove_decrements_size(self):
        m = HashMap()
        m.put("a", 1)
        m.put("b", 2)
        self.assertEqual(m.size(), 2)
        m.remove("a")
        self.assertEqual(m.size(), 1)

    def test_get_after_remove_raises(self):
        m = HashMap()
        m.put("key", "value")
        m.remove("key")
        with self.assertRaises(KeyError):
            m.get("key")


class TestHashMapContains(unittest.TestCase):
    def test_contains_existing_key(self):
        m = HashMap()
        m.put("key", "value")
        self.assertTrue(m.contains("key"))

    def test_contains_nonexistent_key(self):
        m = HashMap()
        self.assertFalse(m.contains("missing"))

    def test_contains_after_remove(self):
        m = HashMap()
        m.put("key", "value")
        m.remove("key")
        self.assertFalse(m.contains("key"))


class TestHashMapClear(unittest.TestCase):
    def test_clear_makes_map_empty(self):
        m = HashMap()
        m.put("a", 1)
        m.put("b", 2)
        m.clear()
        self.assertEqual(m.size(), 0)
        self.assertTrue(m.is_empty())

    def test_clear_empty_map(self):
        m = HashMap()
        m.clear()
        self.assertEqual(m.size(), 0)


class TestHashMapKeysValues(unittest.TestCase):
    def test_keys_returns_all_keys(self):
        m = HashMap()
        m.put("a", 1)
        m.put("b", 2)
        m.put("c", 3)
        keys = m.keys()
        self.assertEqual(sorted(keys), ["a", "b", "c"])

    def test_values_returns_all_values(self):
        m = HashMap()
        m.put("a", 1)
        m.put("b", 2)
        m.put("c", 3)
        values = m.values()
        self.assertEqual(sorted(values), [1, 2, 3])

    def test_keys_values_consistent(self):
        m = HashMap()
        m.put("a", 1)
        m.put("b", 2)
        keys = m.keys()
        values = m.values()
        self.assertEqual(len(keys), len(values))
        for k in keys:
            self.assertIn(m.get(k), values)


class TestHashMapCollisions(unittest.TestCase):
    def test_collision_handling(self):
        m = HashMap(2)
        m.put("a", 1)
        m.put("b", 2)
        m.put("c", 3)
        m.put("d", 4)
        self.assertEqual(m.get("a"), 1)
        self.assertEqual(m.get("b"), 2)
        self.assertEqual(m.get("c"), 3)
        self.assertEqual(m.get("d"), 4)

    def test_remove_with_collisions(self):
        m = HashMap(2)
        m.put("a", 1)
        m.put("b", 2)
        m.put("c", 3)
        m.remove("b")
        self.assertEqual(m.get("a"), 1)
        self.assertEqual(m.get("c"), 3)
        self.assertFalse(m.contains("b"))


class TestHashMapRehashing(unittest.TestCase):
    def test_map_grows_on_load_factor(self):
        m = HashMap(4)
        initial_capacity = m._capacity
        for i in range(10):
            m.put(f"key{i}", i)
        self.assertGreater(m._capacity, initial_capacity)

    def test_all_entries_accessible_after_rehash(self):
        m = HashMap(4)
        for i in range(20):
            m.put(f"key{i}", i)
        for i in range(20):
            self.assertEqual(m.get(f"key{i}"), i)

    def test_size_unchanged_after_rehash(self):
        m = HashMap(4)
        for i in range(10):
            m.put(f"key{i}", i)
        self.assertEqual(m.size(), 10)


class TestHashMapCopy(unittest.TestCase):
    def test_clone_creates_independent_copy(self):
        m = HashMap()
        m.put("a", 1)
        m.put("b", 2)
        clone = m.copy()
        self.assertEqual(clone.get("a"), 1)
        self.assertEqual(clone.get("b"), 2)
        self.assertEqual(clone.size(), 2)

    def test_modify_original_doesnt_affect_clone(self):
        m = HashMap()
        m.put("a", 1)
        clone = m.copy()
        m.put("a", 999)
        m.put("b", 2)
        self.assertEqual(clone.get("a"), 1)
        self.assertFalse(clone.contains("b"))


class TestHashMapTypes(unittest.TestCase):
    def test_string_keys(self):
        m = HashMap()
        m.put("hello", "world")
        self.assertEqual(m.get("hello"), "world")

    def test_string_values(self):
        m = HashMap()
        m.put(1, "one")
        m.put(2, "two")
        self.assertEqual(m.get(1), "one")
        self.assertEqual(m.get(2), "two")

    def test_complex_value_types(self):
        m = HashMap()
        m.put("list", [1, 2, 3])
        m.put("dict", {"nested": "value"})
        self.assertEqual(m.get("list"), [1, 2, 3])
        self.assertEqual(m.get("dict"), {"nested": "value"})


class TestHashMapEdgeCases(unittest.TestCase):
    def test_single_element(self):
        m = HashMap()
        m.put("only", "one")
        self.assertEqual(m.size(), 1)
        self.assertEqual(m.get("only"), "one")

    def test_large_number_of_elements(self):
        m = HashMap()
        for i in range(1000):
            m.put(f"key{i}", i)
        self.assertEqual(m.size(), 1000)
        for i in range(1000):
            self.assertEqual(m.get(f"key{i}"), i)

    def test_many_insertions_and_removals(self):
        m = HashMap()
        for i in range(100):
            m.put(f"key{i}", i)
        for i in range(50):
            m.remove(f"key{i}")
        self.assertEqual(m.size(), 50)
        for i in range(50, 100):
            self.assertEqual(m.get(f"key{i}"), i)


class TestHashMapDunderMethods(unittest.TestCase):
    def test_len(self):
        m = HashMap()
        self.assertEqual(len(m), 0)
        m.put("a", 1)
        self.assertEqual(len(m), 1)

    def test_contains(self):
        m = HashMap()
        m.put("a", 1)
        self.assertTrue("a" in m)
        self.assertFalse("b" in m)

    def test_getitem(self):
        m = HashMap()
        m.put("a", 1)
        self.assertEqual(m["a"], 1)

    def test_getitem_raises(self):
        m = HashMap()
        with self.assertRaises(KeyError):
            _ = m["missing"]

    def test_setitem(self):
        m = HashMap()
        m["a"] = 1
        self.assertEqual(m.get("a"), 1)

    def test_delitem(self):
        m = HashMap()
        m.put("a", 1)
        del m["a"]
        self.assertFalse(m.contains("a"))

    def test_delitem_raises(self):
        m = HashMap()
        with self.assertRaises(KeyError):
            del m["missing"]

    def test_iter(self):
        m = HashMap()
        m.put("a", 1)
        m.put("b", 2)
        m.put("c", 3)
        keys = list(m)
        self.assertEqual(sorted(keys), ["a", "b", "c"])


if __name__ == "__main__":
    unittest.main()
