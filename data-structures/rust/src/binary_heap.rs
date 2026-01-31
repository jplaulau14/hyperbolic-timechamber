#[derive(Debug)]
pub struct BinaryHeap<T> {
    data: Vec<T>,
}

impl<T: Ord> BinaryHeap<T> {
    pub fn new() -> Self {
        BinaryHeap { data: Vec::new() }
    }

    pub fn push(&mut self, value: T) {
        self.data.push(value);
        self.sift_up(self.data.len() - 1);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let last_idx = self.data.len() - 1;
        self.data.swap(0, last_idx);
        let min = self.data.pop();
        if !self.data.is_empty() {
            self.sift_down(0);
        }
        min
    }

    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn from_vec(vec: Vec<T>) -> Self {
        let mut heap = BinaryHeap { data: vec };
        heap.heapify();
        heap
    }

    fn heapify(&mut self) {
        if self.data.len() <= 1 {
            return;
        }
        let last_parent = (self.data.len() - 2) / 2;
        for i in (0..=last_parent).rev() {
            self.sift_down(i);
        }
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx] < self.data[parent] {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        loop {
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;
            let mut smallest = idx;

            if left < self.data.len() && self.data[left] < self.data[smallest] {
                smallest = left;
            }
            if right < self.data.len() && self.data[right] < self.data[smallest] {
                smallest = right;
            }

            if smallest != idx {
                self.data.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }
}

impl<T: Ord> Default for BinaryHeap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone> Clone for BinaryHeap<T> {
    fn clone(&self) -> Self {
        BinaryHeap {
            data: self.data.clone(),
        }
    }
}

impl<T: Ord> FromIterator<T> for BinaryHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<T> = iter.into_iter().collect();
        BinaryHeap::from_vec(vec)
    }
}

impl<T: Ord> Extend<T> for BinaryHeap<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_valid_heap<T: Ord>(heap: &BinaryHeap<T>) -> bool {
        for i in 0..heap.data.len() {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < heap.data.len() && heap.data[i] > heap.data[left] {
                return false;
            }
            if right < heap.data.len() && heap.data[i] > heap.data[right] {
                return false;
            }
        }
        true
    }

    #[test]
    fn new_heap_is_empty() {
        let heap: BinaryHeap<i32> = BinaryHeap::new();
        assert_eq!(heap.size(), 0);
        assert!(heap.is_empty());
    }

    #[test]
    fn peek_on_empty_returns_none() {
        let heap: BinaryHeap<i32> = BinaryHeap::new();
        assert!(heap.peek().is_none());
    }

    #[test]
    fn pop_on_empty_returns_none() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        assert!(heap.pop().is_none());
    }

    #[test]
    fn push_single_element() {
        let mut heap = BinaryHeap::new();
        heap.push(42);
        assert_eq!(heap.size(), 1);
        assert_eq!(heap.peek(), Some(&42));
    }

    #[test]
    fn push_multiple_maintains_heap_property() {
        let mut heap = BinaryHeap::new();
        heap.push(30);
        heap.push(10);
        heap.push(20);
        assert!(is_valid_heap(&heap));
        assert_eq!(heap.peek(), Some(&10));
    }

    #[test]
    fn push_ascending_order() {
        let mut heap = BinaryHeap::new();
        for i in 1..=10 {
            heap.push(i);
        }
        assert!(is_valid_heap(&heap));
        assert_eq!(heap.peek(), Some(&1));
    }

    #[test]
    fn push_descending_order() {
        let mut heap = BinaryHeap::new();
        for i in (1..=10).rev() {
            heap.push(i);
        }
        assert!(is_valid_heap(&heap));
        assert_eq!(heap.peek(), Some(&1));
    }

    #[test]
    fn push_random_order() {
        let mut heap = BinaryHeap::new();
        let values = [5, 3, 8, 1, 9, 2, 7, 4, 6];
        for v in values {
            heap.push(v);
        }
        assert!(is_valid_heap(&heap));
        assert_eq!(heap.peek(), Some(&1));
    }

    #[test]
    fn pop_returns_minimum() {
        let mut heap = BinaryHeap::new();
        heap.push(30);
        heap.push(10);
        heap.push(20);
        assert_eq!(heap.pop(), Some(10));
    }

    #[test]
    fn pop_restores_heap_property() {
        let mut heap = BinaryHeap::new();
        heap.push(30);
        heap.push(10);
        heap.push(20);
        heap.push(5);
        heap.pop();
        assert!(is_valid_heap(&heap));
    }

    #[test]
    fn pop_all_yields_sorted_order() {
        let mut heap = BinaryHeap::new();
        let values = [5, 3, 8, 1, 9, 2, 7, 4, 6];
        for v in values {
            heap.push(v);
        }
        let mut sorted = Vec::new();
        while let Some(v) = heap.pop() {
            sorted.push(v);
        }
        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn size_decrements_after_pop() {
        let mut heap = BinaryHeap::new();
        heap.push(1);
        heap.push(2);
        heap.push(3);
        assert_eq!(heap.size(), 3);
        heap.pop();
        assert_eq!(heap.size(), 2);
    }

    #[test]
    fn peek_returns_minimum_without_removing() {
        let mut heap = BinaryHeap::new();
        heap.push(30);
        heap.push(10);
        heap.push(20);
        assert_eq!(heap.peek(), Some(&10));
        assert_eq!(heap.size(), 3);
    }

    #[test]
    fn multiple_peeks_return_same_value() {
        let mut heap = BinaryHeap::new();
        heap.push(42);
        assert_eq!(heap.peek(), Some(&42));
        assert_eq!(heap.peek(), Some(&42));
        assert_eq!(heap.peek(), Some(&42));
    }

    #[test]
    fn peek_after_push_shows_new_min() {
        let mut heap = BinaryHeap::new();
        heap.push(10);
        assert_eq!(heap.peek(), Some(&10));
        heap.push(5);
        assert_eq!(heap.peek(), Some(&5));
        heap.push(15);
        assert_eq!(heap.peek(), Some(&5));
    }

    #[test]
    fn from_vec_builds_heap() {
        let heap = BinaryHeap::from_vec(vec![5, 3, 8, 1, 9, 2]);
        assert!(is_valid_heap(&heap));
        assert_eq!(heap.peek(), Some(&1));
    }

    #[test]
    fn from_vec_correct_min_at_top() {
        let heap = BinaryHeap::from_vec(vec![100, 50, 200, 25, 75]);
        assert_eq!(heap.peek(), Some(&25));
    }

    #[test]
    fn from_vec_all_elements_present() {
        let values = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
        let mut heap = BinaryHeap::from_vec(values.clone());
        let mut sorted = Vec::new();
        while let Some(v) = heap.pop() {
            sorted.push(v);
        }
        let mut expected = values;
        expected.sort();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn clear_makes_heap_empty() {
        let mut heap = BinaryHeap::new();
        heap.push(1);
        heap.push(2);
        heap.push(3);
        heap.clear();
        assert!(heap.is_empty());
        assert_eq!(heap.size(), 0);
    }

    #[test]
    fn clear_on_empty_is_noop() {
        let mut heap: BinaryHeap<i32> = BinaryHeap::new();
        heap.clear();
        assert!(heap.is_empty());
    }

    #[test]
    fn heap_property_after_operations() {
        let mut heap = BinaryHeap::new();
        for i in [5, 3, 8, 1, 9, 2, 7, 4, 6] {
            heap.push(i);
            assert!(is_valid_heap(&heap));
        }
        for _ in 0..5 {
            heap.pop();
            assert!(is_valid_heap(&heap));
        }
    }

    #[test]
    fn heap_with_duplicates() {
        let mut heap = BinaryHeap::new();
        heap.push(5);
        heap.push(5);
        heap.push(3);
        heap.push(3);
        heap.push(7);
        assert!(is_valid_heap(&heap));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(5));
    }

    #[test]
    fn heap_with_negative_numbers() {
        let mut heap = BinaryHeap::new();
        heap.push(-5);
        heap.push(10);
        heap.push(-20);
        heap.push(0);
        assert!(is_valid_heap(&heap));
        assert_eq!(heap.pop(), Some(-20));
        assert_eq!(heap.pop(), Some(-5));
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut heap = BinaryHeap::new();
        heap.push(10);
        heap.push(5);
        heap.push(15);
        let heap2 = heap.clone();
        assert_eq!(heap.peek(), heap2.peek());
        assert_eq!(heap.size(), heap2.size());
    }

    #[test]
    fn push_to_original_doesnt_affect_clone() {
        let mut heap = BinaryHeap::new();
        heap.push(10);
        let heap2 = heap.clone();
        heap.push(5);
        assert_eq!(heap.size(), 2);
        assert_eq!(heap2.size(), 1);
    }

    #[test]
    fn pop_from_original_doesnt_affect_clone() {
        let mut heap = BinaryHeap::new();
        heap.push(10);
        heap.push(5);
        let heap2 = heap.clone();
        heap.pop();
        assert_eq!(heap.size(), 1);
        assert_eq!(heap2.size(), 2);
    }

    #[test]
    fn works_with_floats() {
        let mut heap = BinaryHeap::new();
        heap.push(3.14f64.to_bits());
        heap.push(1.41f64.to_bits());
        heap.push(2.71f64.to_bits());
        assert!(is_valid_heap(&heap));
    }

    #[test]
    fn works_with_custom_ord_type() {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
        struct Priority(i32);

        let mut heap = BinaryHeap::new();
        heap.push(Priority(30));
        heap.push(Priority(10));
        heap.push(Priority(20));
        assert_eq!(heap.pop(), Some(Priority(10)));
    }

    #[test]
    fn single_element_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(42);
        assert_eq!(heap.peek(), Some(&42));
        assert_eq!(heap.pop(), Some(42));
        assert!(heap.is_empty());
    }

    #[test]
    fn two_element_heap() {
        let mut heap = BinaryHeap::new();
        heap.push(20);
        heap.push(10);
        assert_eq!(heap.pop(), Some(10));
        assert_eq!(heap.pop(), Some(20));
        assert!(heap.is_empty());
    }

    #[test]
    fn large_number_of_elements() {
        let mut heap = BinaryHeap::new();
        for i in (0..1000).rev() {
            heap.push(i);
        }
        assert_eq!(heap.size(), 1000);
        assert!(is_valid_heap(&heap));
        for i in 0..1000 {
            assert_eq!(heap.pop(), Some(i));
        }
    }

    #[test]
    fn many_push_pop_cycles() {
        let mut heap = BinaryHeap::new();
        for _ in 0..100 {
            heap.push(10);
            heap.push(5);
            heap.push(15);
            assert_eq!(heap.pop(), Some(5));
            assert_eq!(heap.pop(), Some(10));
            heap.push(1);
            assert_eq!(heap.pop(), Some(1));
            assert_eq!(heap.pop(), Some(15));
        }
        assert!(heap.is_empty());
    }

    #[test]
    fn default_creates_empty_heap() {
        let heap: BinaryHeap<i32> = BinaryHeap::default();
        assert!(heap.is_empty());
    }

    #[test]
    fn from_iterator() {
        let heap: BinaryHeap<i32> = vec![5, 3, 8, 1, 9].into_iter().collect();
        assert!(is_valid_heap(&heap));
        assert_eq!(heap.peek(), Some(&1));
    }

    #[test]
    fn from_empty_iterator() {
        let heap: BinaryHeap<i32> = Vec::new().into_iter().collect();
        assert!(heap.is_empty());
    }

    #[test]
    fn from_vec_empty() {
        let heap: BinaryHeap<i32> = BinaryHeap::from_vec(vec![]);
        assert!(heap.is_empty());
    }

    #[test]
    fn from_vec_single_element() {
        let heap = BinaryHeap::from_vec(vec![42]);
        assert_eq!(heap.peek(), Some(&42));
        assert_eq!(heap.size(), 1);
    }
}
