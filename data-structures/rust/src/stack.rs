pub struct Stack<T> {
    data: Vec<T>,
}

impl<T> Stack<T> {
    pub fn new() -> Self {
        Stack { data: Vec::new() }
    }

    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    pub fn top(&self) -> Option<&T> {
        self.data.last()
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
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> Clone for Stack<T> {
    fn clone(&self) -> Self {
        Stack {
            data: self.data.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_stack_is_empty() {
        let s: Stack<i32> = Stack::new();
        assert_eq!(s.size(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn top_on_empty_returns_none() {
        let s: Stack<i32> = Stack::new();
        assert!(s.top().is_none());
    }

    #[test]
    fn pop_on_empty_returns_none() {
        let mut s: Stack<i32> = Stack::new();
        assert!(s.pop().is_none());
    }

    #[test]
    fn push_single_element() {
        let mut s = Stack::new();
        s.push(42);
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn push_multiple_elements() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.push(3);
        assert_eq!(s.size(), 3);
    }

    #[test]
    fn push_many_elements() {
        let mut s = Stack::new();
        for i in 0..1000 {
            s.push(i);
        }
        assert_eq!(s.size(), 1000);
    }

    #[test]
    fn pop_returns_top_element() {
        let mut s = Stack::new();
        s.push(10);
        s.push(20);
        assert_eq!(s.pop(), Some(20));
    }

    #[test]
    fn pop_decrements_size() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.pop();
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn pop_all_elements_until_empty() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.push(3);
        s.pop();
        s.pop();
        s.pop();
        assert!(s.is_empty());
    }

    #[test]
    fn lifo_order() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.push(3);
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
    }

    #[test]
    fn top_returns_without_removing() {
        let mut s = Stack::new();
        s.push(42);
        assert_eq!(s.top(), Some(&42));
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn top_after_push_shows_new_element() {
        let mut s = Stack::new();
        s.push(1);
        assert_eq!(s.top(), Some(&1));
        s.push(2);
        assert_eq!(s.top(), Some(&2));
    }

    #[test]
    fn top_multiple_times_returns_same() {
        let mut s = Stack::new();
        s.push(99);
        assert_eq!(s.top(), Some(&99));
        assert_eq!(s.top(), Some(&99));
        assert_eq!(s.top(), Some(&99));
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn clear_makes_stack_empty() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.push(3);
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn clear_on_empty_is_noop() {
        let mut s: Stack<i32> = Stack::new();
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn push_after_clear() {
        let mut s = Stack::new();
        s.push(1);
        s.clear();
        s.push(2);
        assert_eq!(s.size(), 1);
        assert_eq!(s.top(), Some(&2));
    }

    #[test]
    fn size_after_pushes() {
        let mut s = Stack::new();
        for i in 0..5 {
            s.push(i);
            assert_eq!(s.size(), i + 1);
        }
    }

    #[test]
    fn size_after_pops() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.push(3);
        s.pop();
        assert_eq!(s.size(), 2);
        s.pop();
        assert_eq!(s.size(), 1);
    }

    #[test]
    fn is_empty_true_only_when_size_zero() {
        let mut s = Stack::new();
        assert!(s.is_empty());
        s.push(1);
        assert!(!s.is_empty());
        s.pop();
        assert!(s.is_empty());
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        let s2 = s.clone();
        assert_eq!(s.size(), s2.size());
        assert_eq!(s.top(), s2.top());
    }

    #[test]
    fn push_to_original_doesnt_affect_clone() {
        let mut s = Stack::new();
        s.push(1);
        let s2 = s.clone();
        s.push(2);
        assert_eq!(s.size(), 2);
        assert_eq!(s2.size(), 1);
    }

    #[test]
    fn pop_from_original_doesnt_affect_clone() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        let s2 = s.clone();
        s.pop();
        assert_eq!(s.size(), 1);
        assert_eq!(s2.size(), 2);
    }

    #[test]
    fn works_with_strings() {
        let mut s = Stack::new();
        s.push(String::from("hello"));
        s.push(String::from("world"));
        assert_eq!(s.pop(), Some(String::from("world")));
        assert_eq!(s.top(), Some(&String::from("hello")));
    }

    #[test]
    fn works_with_structs() {
        #[derive(Debug, Clone, PartialEq)]
        struct Point {
            x: i32,
            y: i32,
        }
        let mut s = Stack::new();
        s.push(Point { x: 1, y: 2 });
        s.push(Point { x: 3, y: 4 });
        assert_eq!(s.pop(), Some(Point { x: 3, y: 4 }));
    }

    #[test]
    fn single_element_push_pop() {
        let mut s = Stack::new();
        s.push(42);
        assert_eq!(s.pop(), Some(42));
        assert!(s.is_empty());
    }

    #[test]
    fn alternating_push_pop() {
        let mut s = Stack::new();
        s.push(1);
        assert_eq!(s.pop(), Some(1));
        s.push(2);
        s.push(3);
        assert_eq!(s.pop(), Some(3));
        s.push(4);
        assert_eq!(s.pop(), Some(4));
        assert_eq!(s.pop(), Some(2));
        assert!(s.is_empty());
    }

    #[test]
    fn large_number_of_elements() {
        let mut s = Stack::new();
        for i in 0..10000 {
            s.push(i);
        }
        assert_eq!(s.size(), 10000);
        for i in (0..10000).rev() {
            assert_eq!(s.pop(), Some(i));
        }
        assert!(s.is_empty());
    }

    #[test]
    fn default_creates_empty_stack() {
        let s: Stack<i32> = Stack::default();
        assert!(s.is_empty());
    }
}
