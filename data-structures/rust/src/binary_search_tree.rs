use std::cmp::Ordering;

#[derive(Debug)]
struct Node<T> {
    value: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
}

impl<T> Node<T> {
    fn new(value: T) -> Self {
        Node {
            value,
            left: None,
            right: None,
        }
    }
}

#[derive(Debug)]
pub struct BinarySearchTree<T> {
    root: Option<Box<Node<T>>>,
    size: usize,
}

impl<T: Ord> BinarySearchTree<T> {
    pub fn new() -> Self {
        BinarySearchTree {
            root: None,
            size: 0,
        }
    }

    pub fn insert(&mut self, value: T) {
        self.root = Self::insert_node(self.root.take(), value, &mut self.size);
    }

    pub fn remove(&mut self, value: &T) {
        self.root = Self::remove_node(self.root.take(), value, &mut self.size);
    }

    pub fn contains(&self, value: &T) -> bool {
        Self::find_node(&self.root, value).is_some()
    }

    pub fn min(&self) -> Option<&T> {
        self.root.as_deref().map(|n| &Self::find_min(n).value)
    }

    pub fn max(&self) -> Option<&T> {
        self.root.as_deref().map(|n| &Self::find_max(n).value)
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn clear(&mut self) {
        self.root = None;
        self.size = 0;
    }

    pub fn in_order(&self) -> Vec<&T> {
        let mut result = Vec::new();
        Self::in_order_traverse(&self.root, &mut result);
        result
    }

    pub fn pre_order(&self) -> Vec<&T> {
        let mut result = Vec::new();
        Self::pre_order_traverse(&self.root, &mut result);
        result
    }

    pub fn post_order(&self) -> Vec<&T> {
        let mut result = Vec::new();
        Self::post_order_traverse(&self.root, &mut result);
        result
    }

    fn insert_node(node: Option<Box<Node<T>>>, value: T, size: &mut usize) -> Option<Box<Node<T>>> {
        match node {
            None => {
                *size += 1;
                Some(Box::new(Node::new(value)))
            }
            Some(mut n) => {
                match value.cmp(&n.value) {
                    Ordering::Less => n.left = Self::insert_node(n.left.take(), value, size),
                    Ordering::Greater => n.right = Self::insert_node(n.right.take(), value, size),
                    Ordering::Equal => {}
                }
                Some(n)
            }
        }
    }

    fn remove_node(node: Option<Box<Node<T>>>, value: &T, size: &mut usize) -> Option<Box<Node<T>>> {
        match node {
            None => None,
            Some(mut n) => match value.cmp(&n.value) {
                Ordering::Less => {
                    n.left = Self::remove_node(n.left.take(), value, size);
                    Some(n)
                }
                Ordering::Greater => {
                    n.right = Self::remove_node(n.right.take(), value, size);
                    Some(n)
                }
                Ordering::Equal => {
                    *size -= 1;
                    match (n.left.take(), n.right.take()) {
                        (None, None) => None,
                        (Some(left), None) => Some(left),
                        (None, Some(right)) => Some(right),
                        (Some(left), Some(right)) => {
                            *size += 1;
                            let (new_right, successor_value) = Self::remove_min(right);
                            Some(Box::new(Node {
                                value: successor_value,
                                left: Some(left),
                                right: new_right,
                            }))
                        }
                    }
                }
            },
        }
    }

    fn remove_min(node: Box<Node<T>>) -> (Option<Box<Node<T>>>, T) {
        let mut n = node;
        if n.left.is_none() {
            (n.right.take(), n.value)
        } else {
            let (new_left, min_value) = Self::remove_min(n.left.take().unwrap());
            n.left = new_left;
            (Some(n), min_value)
        }
    }

    fn find_node<'a>(node: &'a Option<Box<Node<T>>>, value: &T) -> Option<&'a Node<T>> {
        let mut current = node;
        while let Some(n) = current {
            match value.cmp(&n.value) {
                Ordering::Less => current = &n.left,
                Ordering::Greater => current = &n.right,
                Ordering::Equal => return Some(n),
            }
        }
        None
    }

    fn find_min(node: &Node<T>) -> &Node<T> {
        let mut current = node;
        while let Some(ref left) = current.left {
            current = left;
        }
        current
    }

    fn find_max(node: &Node<T>) -> &Node<T> {
        let mut current = node;
        while let Some(ref right) = current.right {
            current = right;
        }
        current
    }

    fn in_order_traverse<'a>(node: &'a Option<Box<Node<T>>>, result: &mut Vec<&'a T>) {
        if let Some(n) = node {
            Self::in_order_traverse(&n.left, result);
            result.push(&n.value);
            Self::in_order_traverse(&n.right, result);
        }
    }

    fn pre_order_traverse<'a>(node: &'a Option<Box<Node<T>>>, result: &mut Vec<&'a T>) {
        if let Some(n) = node {
            result.push(&n.value);
            Self::pre_order_traverse(&n.left, result);
            Self::pre_order_traverse(&n.right, result);
        }
    }

    fn post_order_traverse<'a>(node: &'a Option<Box<Node<T>>>, result: &mut Vec<&'a T>) {
        if let Some(n) = node {
            Self::post_order_traverse(&n.left, result);
            Self::post_order_traverse(&n.right, result);
            result.push(&n.value);
        }
    }
}

impl<T: Ord> Default for BinarySearchTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone> Clone for BinarySearchTree<T> {
    fn clone(&self) -> Self {
        let mut new_tree = BinarySearchTree::new();
        for value in self.pre_order() {
            new_tree.insert(value.clone());
        }
        new_tree
    }
}

impl<T: Ord> IntoIterator for BinarySearchTree<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        fn collect_in_order<T>(node: Option<Box<Node<T>>>, result: &mut Vec<T>) {
            if let Some(n) = node {
                collect_in_order(n.left, result);
                result.push(n.value);
                collect_in_order(n.right, result);
            }
        }
        let mut values = Vec::with_capacity(self.size);
        collect_in_order(self.root, &mut values);
        values.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_sorted<T: Ord>(v: &[&T]) -> bool {
        v.windows(2).all(|w| w[0] <= w[1])
    }

    fn verify_bst_property<T: Ord>(bst: &BinarySearchTree<T>) -> bool {
        is_sorted(&bst.in_order())
    }

    #[test]
    fn new_tree_is_empty() {
        let bst: BinarySearchTree<i32> = BinarySearchTree::new();
        assert_eq!(bst.size(), 0);
        assert!(bst.is_empty());
    }

    #[test]
    fn min_on_empty_returns_none() {
        let bst: BinarySearchTree<i32> = BinarySearchTree::new();
        assert!(bst.min().is_none());
    }

    #[test]
    fn max_on_empty_returns_none() {
        let bst: BinarySearchTree<i32> = BinarySearchTree::new();
        assert!(bst.max().is_none());
    }

    #[test]
    fn insert_single_element() {
        let mut bst = BinarySearchTree::new();
        bst.insert(42);
        assert_eq!(bst.size(), 1);
        assert!(!bst.is_empty());
        assert!(bst.contains(&42));
    }

    #[test]
    fn insert_multiple_elements() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        assert_eq!(bst.size(), 3);
        assert!(bst.contains(&50));
        assert!(bst.contains(&30));
        assert!(bst.contains(&70));
    }

    #[test]
    fn insert_duplicate_is_noop() {
        let mut bst = BinarySearchTree::new();
        bst.insert(42);
        bst.insert(42);
        assert_eq!(bst.size(), 1);
    }

    #[test]
    fn insert_maintains_bst_property() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        bst.insert(40);
        assert!(verify_bst_property(&bst));
    }

    #[test]
    fn contains_returns_true_for_existing() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        assert!(bst.contains(&50));
        assert!(bst.contains(&30));
        assert!(bst.contains(&70));
    }

    #[test]
    fn contains_returns_false_for_nonexistent() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        assert!(!bst.contains(&30));
        assert!(!bst.contains(&70));
    }

    #[test]
    fn contains_after_insert() {
        let mut bst = BinarySearchTree::new();
        assert!(!bst.contains(&42));
        bst.insert(42);
        assert!(bst.contains(&42));
    }

    #[test]
    fn contains_after_remove() {
        let mut bst = BinarySearchTree::new();
        bst.insert(42);
        assert!(bst.contains(&42));
        bst.remove(&42);
        assert!(!bst.contains(&42));
    }

    #[test]
    fn remove_leaf_node() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.remove(&30);
        assert!(!bst.contains(&30));
        assert!(bst.contains(&50));
        assert!(bst.contains(&70));
        assert_eq!(bst.size(), 2);
    }

    #[test]
    fn remove_node_with_left_child() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(20);
        bst.remove(&30);
        assert!(!bst.contains(&30));
        assert!(bst.contains(&50));
        assert!(bst.contains(&20));
        assert!(verify_bst_property(&bst));
    }

    #[test]
    fn remove_node_with_right_child() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(40);
        bst.remove(&30);
        assert!(!bst.contains(&30));
        assert!(bst.contains(&50));
        assert!(bst.contains(&40));
        assert!(verify_bst_property(&bst));
    }

    #[test]
    fn remove_node_with_two_children() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        bst.insert(40);
        bst.remove(&30);
        assert!(!bst.contains(&30));
        assert!(bst.contains(&50));
        assert!(bst.contains(&70));
        assert!(bst.contains(&20));
        assert!(bst.contains(&40));
        assert!(verify_bst_property(&bst));
    }

    #[test]
    fn remove_root_node() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.remove(&50);
        assert!(!bst.contains(&50));
        assert!(bst.contains(&30));
        assert!(bst.contains(&70));
        assert!(verify_bst_property(&bst));
    }

    #[test]
    fn remove_nonexistent_is_noop() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.remove(&100);
        assert_eq!(bst.size(), 1);
        assert!(bst.contains(&50));
    }

    #[test]
    fn size_decrements_after_remove() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        assert_eq!(bst.size(), 3);
        bst.remove(&30);
        assert_eq!(bst.size(), 2);
    }

    #[test]
    fn min_returns_smallest() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        assert_eq!(bst.min(), Some(&20));
    }

    #[test]
    fn max_returns_largest() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(80);
        assert_eq!(bst.max(), Some(&80));
    }

    #[test]
    fn min_max_after_insert() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        assert_eq!(bst.min(), Some(&50));
        assert_eq!(bst.max(), Some(&50));
        bst.insert(30);
        assert_eq!(bst.min(), Some(&30));
        bst.insert(70);
        assert_eq!(bst.max(), Some(&70));
    }

    #[test]
    fn min_max_after_remove() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.remove(&30);
        assert_eq!(bst.min(), Some(&50));
        bst.remove(&70);
        assert_eq!(bst.max(), Some(&50));
    }

    #[test]
    fn in_order_yields_sorted() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        bst.insert(40);
        assert_eq!(bst.in_order(), vec![&20, &30, &40, &50, &70]);
    }

    #[test]
    fn pre_order_yields_correct_sequence() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        bst.insert(40);
        assert_eq!(bst.pre_order(), vec![&50, &30, &20, &40, &70]);
    }

    #[test]
    fn post_order_yields_correct_sequence() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        bst.insert(40);
        assert_eq!(bst.post_order(), vec![&20, &40, &30, &70, &50]);
    }

    #[test]
    fn traversals_on_empty_tree() {
        let bst: BinarySearchTree<i32> = BinarySearchTree::new();
        assert!(bst.in_order().is_empty());
        assert!(bst.pre_order().is_empty());
        assert!(bst.post_order().is_empty());
    }

    #[test]
    fn clear_makes_tree_empty() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.clear();
        assert!(bst.is_empty());
        assert_eq!(bst.size(), 0);
    }

    #[test]
    fn clear_on_empty_is_noop() {
        let mut bst: BinarySearchTree<i32> = BinarySearchTree::new();
        bst.clear();
        assert!(bst.is_empty());
    }

    #[test]
    fn bst_property_holds_after_operations() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        bst.insert(40);
        assert!(verify_bst_property(&bst));
        bst.remove(&30);
        assert!(verify_bst_property(&bst));
        bst.insert(35);
        assert!(verify_bst_property(&bst));
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        let clone = bst.clone();
        assert_eq!(clone.size(), 3);
        assert!(clone.contains(&50));
        assert!(clone.contains(&30));
        assert!(clone.contains(&70));
    }

    #[test]
    fn insert_to_original_doesnt_affect_clone() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        let clone = bst.clone();
        bst.insert(30);
        assert_eq!(bst.size(), 2);
        assert_eq!(clone.size(), 1);
        assert!(!clone.contains(&30));
    }

    #[test]
    fn remove_from_original_doesnt_affect_clone() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        let clone = bst.clone();
        bst.remove(&30);
        assert!(!bst.contains(&30));
        assert!(clone.contains(&30));
    }

    #[test]
    fn single_element_tree() {
        let mut bst = BinarySearchTree::new();
        bst.insert(42);
        assert_eq!(bst.min(), Some(&42));
        assert_eq!(bst.max(), Some(&42));
        assert!(bst.contains(&42));
        bst.remove(&42);
        assert!(bst.is_empty());
    }

    #[test]
    fn sorted_insert_degenerates_to_list() {
        let mut bst = BinarySearchTree::new();
        for i in 1..=10 {
            bst.insert(i);
        }
        assert_eq!(bst.size(), 10);
        assert_eq!(bst.min(), Some(&1));
        assert_eq!(bst.max(), Some(&10));
        assert!(verify_bst_property(&bst));
    }

    #[test]
    fn large_number_of_elements() {
        let mut bst = BinarySearchTree::new();
        for i in 1..=1000 {
            bst.insert(i);
        }
        assert_eq!(bst.size(), 1000);
        assert_eq!(bst.min(), Some(&1));
        assert_eq!(bst.max(), Some(&1000));
    }

    #[test]
    fn negative_numbers() {
        let mut bst = BinarySearchTree::new();
        bst.insert(-10);
        bst.insert(0);
        bst.insert(10);
        bst.insert(-20);
        assert_eq!(bst.min(), Some(&-20));
        assert_eq!(bst.max(), Some(&10));
        assert!(bst.contains(&-10));
        assert!(verify_bst_property(&bst));
    }

    #[test]
    fn remove_all_elements_one_by_one() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        bst.insert(40);
        bst.remove(&20);
        bst.remove(&40);
        bst.remove(&30);
        bst.remove(&70);
        bst.remove(&50);
        assert!(bst.is_empty());
    }

    #[test]
    fn default_creates_empty_tree() {
        let bst: BinarySearchTree<i32> = BinarySearchTree::default();
        assert!(bst.is_empty());
    }

    #[test]
    fn into_iter_yields_sorted() {
        let mut bst = BinarySearchTree::new();
        bst.insert(50);
        bst.insert(30);
        bst.insert(70);
        bst.insert(20);
        bst.insert(40);
        let result: Vec<i32> = bst.into_iter().collect();
        assert_eq!(result, vec![20, 30, 40, 50, 70]);
    }

    #[test]
    fn works_with_strings() {
        let mut bst = BinarySearchTree::new();
        bst.insert("banana");
        bst.insert("apple");
        bst.insert("cherry");
        assert_eq!(bst.min(), Some(&"apple"));
        assert_eq!(bst.max(), Some(&"cherry"));
        assert_eq!(bst.in_order(), vec![&"apple", &"banana", &"cherry"]);
    }

    #[test]
    fn into_iter_with_owned_strings() {
        let mut bst = BinarySearchTree::new();
        bst.insert(String::from("banana"));
        bst.insert(String::from("apple"));
        bst.insert(String::from("cherry"));
        let result: Vec<String> = bst.into_iter().collect();
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }
}
