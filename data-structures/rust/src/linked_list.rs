struct Node<T> {
    value: T,
    next: Option<Box<Node<T>>>,
}

pub struct LinkedList<T> {
    head: Option<Box<Node<T>>>,
    tail: *mut Node<T>,
    size: usize,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        LinkedList {
            head: None,
            tail: std::ptr::null_mut(),
            size: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn push_front(&mut self, value: T) {
        let mut new_node = Box::new(Node { value, next: self.head.take() });

        if self.tail.is_null() {
            self.tail = &mut *new_node;
        }

        self.head = Some(new_node);
        self.size += 1;
    }

    pub fn push_back(&mut self, value: T) {
        let mut new_node = Box::new(Node { value, next: None });
        let raw_tail: *mut Node<T> = &mut *new_node;

        if self.tail.is_null() {
            self.head = Some(new_node);
        } else {
            // SAFETY: tail is non-null and points to a valid Node that is owned by this list.
            // The node remains valid because we only update tail when nodes are added/removed.
            unsafe {
                (*self.tail).next = Some(new_node);
            }
        }

        self.tail = raw_tail;
        self.size += 1;
    }

    pub fn pop_front(&mut self) -> Option<T> {
        self.head.take().map(|node| {
            self.head = node.next;
            self.size -= 1;

            if self.head.is_none() {
                self.tail = std::ptr::null_mut();
            }

            node.value
        })
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.head.is_none() {
            return None;
        }

        self.size -= 1;

        if self.head.as_ref().unwrap().next.is_none() {
            self.tail = std::ptr::null_mut();
            return self.head.take().map(|node| node.value);
        }

        let mut current = self.head.as_mut().unwrap();
        while current.next.as_ref().unwrap().next.is_some() {
            current = current.next.as_mut().unwrap();
        }

        self.tail = &mut **current;
        current.next.take().map(|node| node.value)
    }

    pub fn front(&self) -> Option<&T> {
        self.head.as_ref().map(|node| &node.value)
    }

    pub fn back(&self) -> Option<&T> {
        if self.tail.is_null() {
            None
        } else {
            // SAFETY: tail is non-null and points to a valid Node owned by this list.
            // The reference lifetime is tied to &self, ensuring the node outlives the reference.
            unsafe { Some(&(*self.tail).value) }
        }
    }

    pub fn at(&self, index: usize) -> Option<&T> {
        if index >= self.size {
            return None;
        }

        let mut current = self.head.as_ref();
        for _ in 0..index {
            current = current.unwrap().next.as_ref();
        }

        current.map(|node| &node.value)
    }

    pub fn insert_at(&mut self, index: usize, value: T) -> bool {
        if index > self.size {
            return false;
        }

        if index == 0 {
            self.push_front(value);
            return true;
        }

        if index == self.size {
            self.push_back(value);
            return true;
        }

        let mut current = self.head.as_mut().unwrap();
        for _ in 0..index - 1 {
            current = current.next.as_mut().unwrap();
        }

        let new_node = Box::new(Node {
            value,
            next: current.next.take(),
        });
        current.next = Some(new_node);
        self.size += 1;
        true
    }

    pub fn remove_at(&mut self, index: usize) -> Option<T> {
        if index >= self.size {
            return None;
        }

        if index == 0 {
            return self.pop_front();
        }

        let mut current = self.head.as_mut().unwrap();
        for _ in 0..index - 1 {
            current = current.next.as_mut().unwrap();
        }

        let removed = current.next.take();
        if let Some(mut node) = removed {
            current.next = node.next.take();
            self.size -= 1;

            if current.next.is_none() {
                self.tail = &mut **current;
            }

            return Some(node.value);
        }

        None
    }

    pub fn clear(&mut self) {
        while self.pop_front().is_some() {}
    }

    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            current: self.head.as_deref(),
        }
    }
}

impl<T> Default for LinkedList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: Clone> Clone for LinkedList<T> {
    fn clone(&self) -> Self {
        let mut new_list = LinkedList::new();
        for value in self.iter() {
            new_list.push_back(value.clone());
        }
        new_list
    }
}

pub struct Iter<'a, T> {
    current: Option<&'a Node<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.current.map(|node| {
            self.current = node.next.as_deref();
            &node.value
        })
    }
}

pub struct IntoIter<T> {
    list: LinkedList<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.list.pop_front()
    }
}

impl<T> IntoIterator for LinkedList<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { list: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_list_is_empty() {
        let list: LinkedList<i32> = LinkedList::new();
        assert_eq!(list.size(), 0);
        assert!(list.is_empty());
    }

    #[test]
    fn front_back_on_empty_list() {
        let list: LinkedList<i32> = LinkedList::new();
        assert!(list.front().is_none());
        assert!(list.back().is_none());
    }

    #[test]
    fn push_front_single_element() {
        let mut list = LinkedList::new();
        list.push_front(42);
        assert_eq!(list.size(), 1);
        assert_eq!(list.front(), Some(&42));
        assert_eq!(list.back(), Some(&42));
    }

    #[test]
    fn push_front_multiple_elements() {
        let mut list = LinkedList::new();
        list.push_front(1);
        list.push_front(2);
        list.push_front(3);
        assert_eq!(list.front(), Some(&3));
        assert_eq!(list.back(), Some(&1));

        let values: Vec<_> = list.iter().cloned().collect();
        assert_eq!(values, vec![3, 2, 1]);
    }

    #[test]
    fn push_back_single_element() {
        let mut list = LinkedList::new();
        list.push_back(42);
        assert_eq!(list.size(), 1);
        assert_eq!(list.front(), Some(&42));
        assert_eq!(list.back(), Some(&42));
    }

    #[test]
    fn push_back_multiple_elements() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&3));

        let values: Vec<_> = list.iter().cloned().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn mixed_push_front_and_push_back() {
        let mut list = LinkedList::new();
        list.push_back(2);
        list.push_front(1);
        list.push_back(3);
        list.push_front(0);

        let values: Vec<_> = list.iter().cloned().collect();
        assert_eq!(values, vec![0, 1, 2, 3]);
    }

    #[test]
    fn pop_front_returns_correct_value() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.size(), 2);
        assert_eq!(list.front(), Some(&2));
    }

    #[test]
    fn pop_front_until_empty() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);

        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_front(), Some(2));
        assert!(list.is_empty());
        assert!(list.front().is_none());
        assert!(list.back().is_none());
    }

    #[test]
    fn pop_front_on_empty_list() {
        let mut list: LinkedList<i32> = LinkedList::new();
        assert!(list.pop_front().is_none());
    }

    #[test]
    fn pop_back_returns_correct_value() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.size(), 2);
        assert_eq!(list.back(), Some(&2));
    }

    #[test]
    fn pop_back_until_empty() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);

        assert_eq!(list.pop_back(), Some(2));
        assert_eq!(list.pop_back(), Some(1));
        assert!(list.is_empty());
        assert!(list.front().is_none());
        assert!(list.back().is_none());
    }

    #[test]
    fn pop_back_on_empty_list() {
        let mut list: LinkedList<i32> = LinkedList::new();
        assert!(list.pop_back().is_none());
    }

    #[test]
    fn front_returns_first_without_removing() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);

        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.size(), 2);
    }

    #[test]
    fn back_returns_last_without_removing() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);

        assert_eq!(list.back(), Some(&2));
        assert_eq!(list.back(), Some(&2));
        assert_eq!(list.size(), 2);
    }

    #[test]
    fn at_returns_first_element() {
        let mut list = LinkedList::new();
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        assert_eq!(list.at(0), Some(&10));
    }

    #[test]
    fn at_returns_last_element() {
        let mut list = LinkedList::new();
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        assert_eq!(list.at(2), Some(&30));
    }

    #[test]
    fn at_returns_middle_element() {
        let mut list = LinkedList::new();
        list.push_back(10);
        list.push_back(20);
        list.push_back(30);

        assert_eq!(list.at(1), Some(&20));
    }

    #[test]
    fn at_invalid_index_returns_none() {
        let mut list = LinkedList::new();
        list.push_back(1);

        assert!(list.at(1).is_none());
        assert!(list.at(100).is_none());
    }

    #[test]
    fn insert_at_zero_same_as_push_front() {
        let mut list = LinkedList::new();
        list.push_back(2);
        list.push_back(3);
        list.insert_at(0, 1);

        let values: Vec<_> = list.iter().cloned().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn insert_at_size_same_as_push_back() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.insert_at(2, 3);

        let values: Vec<_> = list.iter().cloned().collect();
        assert_eq!(values, vec![1, 2, 3]);
        assert_eq!(list.back(), Some(&3));
    }

    #[test]
    fn insert_at_middle() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(3);
        list.insert_at(1, 2);

        let values: Vec<_> = list.iter().cloned().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn insert_at_invalid_returns_false() {
        let mut list = LinkedList::new();
        list.push_back(1);

        assert!(!list.insert_at(5, 99));
        assert_eq!(list.size(), 1);
    }

    #[test]
    fn remove_at_zero_same_as_pop_front() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.remove_at(0), Some(1));
        assert_eq!(list.front(), Some(&2));
    }

    #[test]
    fn remove_at_last_same_as_pop_back() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.remove_at(2), Some(3));
        assert_eq!(list.back(), Some(&2));
    }

    #[test]
    fn remove_at_middle() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        assert_eq!(list.remove_at(1), Some(2));
        let values: Vec<_> = list.iter().cloned().collect();
        assert_eq!(values, vec![1, 3]);
    }

    #[test]
    fn remove_at_invalid_returns_none() {
        let mut list = LinkedList::new();
        list.push_back(1);

        assert!(list.remove_at(5).is_none());
    }

    #[test]
    fn clear_makes_list_empty() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        list.clear();

        assert!(list.is_empty());
        assert!(list.front().is_none());
        assert!(list.back().is_none());
    }

    #[test]
    fn clear_on_empty_list_is_noop() {
        let mut list: LinkedList<i32> = LinkedList::new();
        list.clear();
        assert!(list.is_empty());
    }

    #[test]
    fn iterate_in_correct_order() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        let values: Vec<_> = list.iter().cloned().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn iterate_empty_list() {
        let list: LinkedList<i32> = LinkedList::new();
        let values: Vec<_> = list.iter().collect();
        assert!(values.is_empty());
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut original = LinkedList::new();
        original.push_back(1);
        original.push_back(2);
        original.push_back(3);

        let cloned = original.clone();

        assert_eq!(cloned.size(), 3);
        let values: Vec<_> = cloned.iter().cloned().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn modifying_original_doesnt_affect_clone() {
        let mut original = LinkedList::new();
        original.push_back(1);
        original.push_back(2);

        let cloned = original.clone();

        original.push_back(3);
        original.pop_front();

        assert_eq!(cloned.size(), 2);
        let values: Vec<_> = cloned.iter().cloned().collect();
        assert_eq!(values, vec![1, 2]);
    }

    #[test]
    fn works_with_strings() {
        let mut list = LinkedList::new();
        list.push_back(String::from("hello"));
        list.push_back(String::from("world"));

        assert_eq!(list.front(), Some(&String::from("hello")));
        assert_eq!(list.back(), Some(&String::from("world")));
        assert_eq!(list.pop_front(), Some(String::from("hello")));
    }

    #[test]
    fn single_element_list() {
        let mut list = LinkedList::new();
        list.push_back(42);

        assert_eq!(list.front(), list.back());
        assert_eq!(list.pop_front(), Some(42));
        assert!(list.is_empty());
    }

    #[test]
    fn two_element_list_linking() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);

        assert_eq!(list.front(), Some(&1));
        assert_eq!(list.back(), Some(&2));
        assert_eq!(list.at(0), Some(&1));
        assert_eq!(list.at(1), Some(&2));
    }

    #[test]
    fn default_trait() {
        let list: LinkedList<i32> = LinkedList::default();
        assert!(list.is_empty());
    }

    #[test]
    fn into_iter() {
        let mut list = LinkedList::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        let values: Vec<_> = list.into_iter().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }
}
