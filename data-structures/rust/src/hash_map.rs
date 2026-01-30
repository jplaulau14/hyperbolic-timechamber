use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

struct Entry<K, V> {
    key: K,
    value: V,
    next: Option<Box<Entry<K, V>>>,
}

pub struct HashMap<K, V> {
    buckets: Vec<Option<Box<Entry<K, V>>>>,
    size: usize,
    capacity: usize,
}

impl<K: Hash + Eq, V> HashMap<K, V> {
    const DEFAULT_CAPACITY: usize = 16;
    const LOAD_FACTOR: f64 = 0.75;

    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let mut buckets = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buckets.push(None);
        }
        HashMap {
            buckets,
            size: 0,
            capacity,
        }
    }

    fn hash(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.capacity
    }

    pub fn insert(&mut self, key: K, value: V) {
        if (self.size + 1) as f64 / self.capacity as f64 > Self::LOAD_FACTOR {
            self.rehash();
        }

        let index = self.hash(&key);
        let bucket = &mut self.buckets[index];

        let mut current = bucket.as_mut();
        while let Some(entry) = current {
            if entry.key == key {
                entry.value = value;
                return;
            }
            current = entry.next.as_mut();
        }

        let new_entry = Box::new(Entry {
            key,
            value,
            next: self.buckets[index].take(),
        });
        self.buckets[index] = Some(new_entry);
        self.size += 1;
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        let index = self.hash(key);
        let mut current = self.buckets[index].as_ref();

        while let Some(entry) = current {
            if entry.key == *key {
                return Some(&entry.value);
            }
            current = entry.next.as_ref();
        }

        None
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let index = self.hash(key);
        let mut current = self.buckets[index].as_mut();

        while let Some(entry) = current {
            if entry.key == *key {
                return Some(&mut entry.value);
            }
            current = entry.next.as_mut();
        }

        None
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        let index = self.hash(key);

        if self.buckets[index].is_none() {
            return None;
        }

        if self.buckets[index].as_ref().unwrap().key == *key {
            let entry = self.buckets[index].take().unwrap();
            self.buckets[index] = entry.next;
            self.size -= 1;
            return Some(entry.value);
        }

        let mut current = self.buckets[index].as_mut().unwrap();
        while current.next.is_some() {
            if current.next.as_ref().unwrap().key == *key {
                let removed = current.next.take().unwrap();
                current.next = removed.next;
                self.size -= 1;
                return Some(removed.value);
            }
            current = current.next.as_mut().unwrap();
        }

        None
    }

    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = None;
        }
        self.size = 0;
    }

    pub fn keys(&self) -> Vec<&K> {
        let mut keys = Vec::with_capacity(self.size);
        for bucket in &self.buckets {
            let mut current = bucket.as_ref();
            while let Some(entry) = current {
                keys.push(&entry.key);
                current = entry.next.as_ref();
            }
        }
        keys
    }

    pub fn values(&self) -> Vec<&V> {
        let mut values = Vec::with_capacity(self.size);
        for bucket in &self.buckets {
            let mut current = bucket.as_ref();
            while let Some(entry) = current {
                values.push(&entry.value);
                current = entry.next.as_ref();
            }
        }
        values
    }

    fn rehash(&mut self) {
        let new_capacity = self.capacity * 2;
        let mut new_buckets = Vec::with_capacity(new_capacity);
        for _ in 0..new_capacity {
            new_buckets.push(None);
        }

        let old_buckets = std::mem::replace(&mut self.buckets, new_buckets);
        self.capacity = new_capacity;

        for mut bucket in old_buckets {
            while let Some(mut entry) = bucket {
                bucket = entry.next.take();
                let index = self.hash(&entry.key);
                entry.next = self.buckets[index].take();
                self.buckets[index] = Some(entry);
            }
        }
    }
}

impl<K: Hash + Eq, V> Default for HashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone, V: Clone> Clone for HashMap<K, V> {
    fn clone(&self) -> Self {
        let mut new_map = HashMap::with_capacity(self.capacity);
        for bucket in &self.buckets {
            let mut current = bucket.as_ref();
            while let Some(entry) = current {
                new_map.insert(entry.key.clone(), entry.value.clone());
                current = entry.next.as_ref();
            }
        }
        new_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_map_is_empty() {
        let map: HashMap<i32, i32> = HashMap::new();
        assert_eq!(map.size(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn default_capacity_is_16() {
        let map: HashMap<i32, i32> = HashMap::new();
        assert_eq!(map.capacity(), 16);
    }

    #[test]
    fn insert_single_pair() {
        let mut map = HashMap::new();
        map.insert("key", 42);
        assert_eq!(map.size(), 1);
        assert_eq!(map.get(&"key"), Some(&42));
    }

    #[test]
    fn insert_multiple_pairs() {
        let mut map = HashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);
        assert_eq!(map.size(), 3);
        assert_eq!(map.get(&"a"), Some(&1));
        assert_eq!(map.get(&"b"), Some(&2));
        assert_eq!(map.get(&"c"), Some(&3));
    }

    #[test]
    fn insert_duplicate_key_updates_value() {
        let mut map = HashMap::new();
        map.insert("key", 1);
        map.insert("key", 2);
        assert_eq!(map.size(), 1);
        assert_eq!(map.get(&"key"), Some(&2));
    }

    #[test]
    fn insert_triggers_rehash() {
        let mut map: HashMap<i32, i32> = HashMap::with_capacity(4);
        map.insert(1, 1);
        map.insert(2, 2);
        map.insert(3, 3);
        assert_eq!(map.capacity(), 4);
        map.insert(4, 4);
        assert_eq!(map.capacity(), 8);
        assert_eq!(map.size(), 4);
    }

    #[test]
    fn get_existing_key() {
        let mut map = HashMap::new();
        map.insert("hello", "world");
        assert_eq!(map.get(&"hello"), Some(&"world"));
    }

    #[test]
    fn get_nonexistent_key() {
        let map: HashMap<&str, i32> = HashMap::new();
        assert_eq!(map.get(&"missing"), None);
    }

    #[test]
    fn get_after_update() {
        let mut map = HashMap::new();
        map.insert("key", "old");
        map.insert("key", "new");
        assert_eq!(map.get(&"key"), Some(&"new"));
    }

    #[test]
    fn remove_existing_key() {
        let mut map = HashMap::new();
        map.insert("key", 42);
        let removed = map.remove(&"key");
        assert_eq!(removed, Some(42));
        assert!(map.is_empty());
    }

    #[test]
    fn remove_nonexistent_key() {
        let mut map: HashMap<&str, i32> = HashMap::new();
        let removed = map.remove(&"missing");
        assert_eq!(removed, None);
    }

    #[test]
    fn remove_decrements_size() {
        let mut map = HashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.remove(&"a");
        assert_eq!(map.size(), 1);
    }

    #[test]
    fn get_after_remove() {
        let mut map = HashMap::new();
        map.insert("key", 42);
        map.remove(&"key");
        assert_eq!(map.get(&"key"), None);
    }

    #[test]
    fn contains_existing_key() {
        let mut map = HashMap::new();
        map.insert("key", 1);
        assert!(map.contains(&"key"));
    }

    #[test]
    fn contains_nonexistent_key() {
        let map: HashMap<&str, i32> = HashMap::new();
        assert!(!map.contains(&"missing"));
    }

    #[test]
    fn contains_after_remove() {
        let mut map = HashMap::new();
        map.insert("key", 1);
        map.remove(&"key");
        assert!(!map.contains(&"key"));
    }

    #[test]
    fn clear_makes_map_empty() {
        let mut map = HashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.get(&"a"), None);
    }

    #[test]
    fn clear_on_empty_map() {
        let mut map: HashMap<i32, i32> = HashMap::new();
        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn keys_returns_all_keys() {
        let mut map = HashMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");
        let mut keys: Vec<_> = map.keys().into_iter().cloned().collect();
        keys.sort();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn values_returns_all_values() {
        let mut map = HashMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");
        let mut values: Vec<_> = map.values().into_iter().cloned().collect();
        values.sort();
        assert_eq!(values, vec!["a", "b", "c"]);
    }

    #[test]
    fn keys_and_values_consistent() {
        let mut map = HashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        let keys = map.keys();
        let values = map.values();
        assert_eq!(keys.len(), values.len());
        for key in keys {
            assert!(map.get(key).is_some());
        }
    }

    #[test]
    fn collision_handling_insert() {
        let mut map: HashMap<i32, &str> = HashMap::with_capacity(1);
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        assert_eq!(map.size(), 3);
    }

    #[test]
    fn collision_handling_get() {
        let mut map: HashMap<i32, &str> = HashMap::with_capacity(1);
        map.insert(1, "one");
        map.insert(2, "two");
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), Some(&"two"));
    }

    #[test]
    fn collision_handling_remove() {
        let mut map: HashMap<i32, &str> = HashMap::with_capacity(1);
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        assert_eq!(map.remove(&2), Some("two"));
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&3), Some(&"three"));
    }

    #[test]
    fn rehash_grows_capacity() {
        let mut map: HashMap<i32, i32> = HashMap::with_capacity(4);
        for i in 0..4 {
            map.insert(i, i);
        }
        assert!(map.capacity() > 4);
    }

    #[test]
    fn all_entries_accessible_after_rehash() {
        let mut map: HashMap<i32, i32> = HashMap::with_capacity(4);
        for i in 0..10 {
            map.insert(i, i * 10);
        }
        for i in 0..10 {
            assert_eq!(map.get(&i), Some(&(i * 10)));
        }
    }

    #[test]
    fn size_unchanged_after_rehash() {
        let mut map: HashMap<i32, i32> = HashMap::with_capacity(4);
        for i in 0..10 {
            map.insert(i, i);
        }
        assert_eq!(map.size(), 10);
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut original = HashMap::new();
        original.insert("a", 1);
        original.insert("b", 2);
        let cloned = original.clone();
        assert_eq!(cloned.size(), 2);
        assert_eq!(cloned.get(&"a"), Some(&1));
        assert_eq!(cloned.get(&"b"), Some(&2));
    }

    #[test]
    fn modify_original_doesnt_affect_clone() {
        let mut original = HashMap::new();
        original.insert("a", 1);
        let cloned = original.clone();
        original.insert("b", 2);
        original.remove(&"a");
        assert_eq!(cloned.size(), 1);
        assert_eq!(cloned.get(&"a"), Some(&1));
        assert_eq!(cloned.get(&"b"), None);
    }

    #[test]
    fn works_with_string_keys() {
        let mut map = HashMap::new();
        map.insert(String::from("hello"), 1);
        map.insert(String::from("world"), 2);
        assert_eq!(map.get(&String::from("hello")), Some(&1));
    }

    #[test]
    fn works_with_string_values() {
        let mut map = HashMap::new();
        map.insert(1, String::from("one"));
        map.insert(2, String::from("two"));
        assert_eq!(map.get(&1), Some(&String::from("one")));
    }

    #[test]
    fn works_with_complex_value_types() {
        let mut map: HashMap<&str, Vec<i32>> = HashMap::new();
        map.insert("numbers", vec![1, 2, 3]);
        assert_eq!(map.get(&"numbers"), Some(&vec![1, 2, 3]));
    }

    #[test]
    fn single_element_map() {
        let mut map = HashMap::new();
        map.insert(42, "answer");
        assert_eq!(map.size(), 1);
        assert_eq!(map.get(&42), Some(&"answer"));
        assert!(map.contains(&42));
    }

    #[test]
    fn large_number_of_elements() {
        let mut map = HashMap::new();
        for i in 0..1000 {
            map.insert(i, i * 2);
        }
        assert_eq!(map.size(), 1000);
        for i in 0..1000 {
            assert_eq!(map.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn many_insertions_and_removals() {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, i);
        }
        for i in 0..50 {
            map.remove(&i);
        }
        assert_eq!(map.size(), 50);
        for i in 0..50 {
            assert!(!map.contains(&i));
        }
        for i in 50..100 {
            assert!(map.contains(&i));
        }
    }

    #[test]
    fn default_trait() {
        let map: HashMap<i32, i32> = HashMap::default();
        assert!(map.is_empty());
        assert_eq!(map.capacity(), 16);
    }

    #[test]
    fn get_mut_modifies_value() {
        let mut map = HashMap::new();
        map.insert("key", 1);
        if let Some(val) = map.get_mut(&"key") {
            *val = 100;
        }
        assert_eq!(map.get(&"key"), Some(&100));
    }

    #[test]
    fn remove_from_middle_of_chain() {
        let mut map: HashMap<i32, i32> = HashMap::with_capacity(1);
        map.insert(1, 1);
        map.insert(2, 2);
        map.insert(3, 3);
        map.remove(&2);
        assert_eq!(map.get(&1), Some(&1));
        assert_eq!(map.get(&2), None);
        assert_eq!(map.get(&3), Some(&3));
    }

    #[test]
    fn remove_first_in_chain() {
        let mut map: HashMap<i32, i32> = HashMap::with_capacity(1);
        map.insert(1, 1);
        map.insert(2, 2);
        map.insert(3, 3);
        let last_inserted = map.keys().iter().find(|&&k| *k == 3).is_some();
        assert!(last_inserted);
        map.remove(&3);
        assert_eq!(map.get(&1), Some(&1));
        assert_eq!(map.get(&2), Some(&2));
    }
}
