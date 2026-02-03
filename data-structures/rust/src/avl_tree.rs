use std::cmp::Ordering;

#[derive(Debug)]
struct Node<T> {
    value: T,
    left: Option<Box<Node<T>>>,
    right: Option<Box<Node<T>>>,
    height: usize,
}

impl<T> Node<T> {
    fn new(value: T) -> Self {
        Node {
            value,
            left: None,
            right: None,
            height: 1,
        }
    }
}

#[derive(Debug)]
pub struct AVLTree<T> {
    root: Option<Box<Node<T>>>,
    size: usize,
}

impl<T: Ord> AVLTree<T> {
    pub fn new() -> Self {
        AVLTree {
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

    pub fn height(&self) -> usize {
        Self::node_height(&self.root)
    }

    pub fn is_balanced(&self) -> bool {
        Self::check_balance(&self.root)
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

    fn node_height(node: &Option<Box<Node<T>>>) -> usize {
        node.as_ref().map_or(0, |n| n.height)
    }

    fn update_height(node: &mut Box<Node<T>>) {
        node.height = 1 + Self::node_height(&node.left).max(Self::node_height(&node.right));
    }

    fn balance_factor(node: &Box<Node<T>>) -> i32 {
        Self::node_height(&node.left) as i32 - Self::node_height(&node.right) as i32
    }

    fn rotate_right(mut y: Box<Node<T>>) -> Box<Node<T>> {
        let mut x = y.left.take().unwrap();
        y.left = x.right.take();
        Self::update_height(&mut y);
        x.right = Some(y);
        Self::update_height(&mut x);
        x
    }

    fn rotate_left(mut x: Box<Node<T>>) -> Box<Node<T>> {
        let mut y = x.right.take().unwrap();
        x.right = y.left.take();
        Self::update_height(&mut x);
        y.left = Some(x);
        Self::update_height(&mut y);
        y
    }

    fn rebalance(mut node: Box<Node<T>>) -> Box<Node<T>> {
        Self::update_height(&mut node);
        let balance = Self::balance_factor(&node);

        if balance > 1 {
            if Self::balance_factor(node.left.as_ref().unwrap()) < 0 {
                node.left = Some(Self::rotate_left(node.left.take().unwrap()));
            }
            return Self::rotate_right(node);
        }

        if balance < -1 {
            if Self::balance_factor(node.right.as_ref().unwrap()) > 0 {
                node.right = Some(Self::rotate_right(node.right.take().unwrap()));
            }
            return Self::rotate_left(node);
        }

        node
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
                    Ordering::Equal => return Some(n),
                }
                Some(Self::rebalance(n))
            }
        }
    }

    fn remove_node(node: Option<Box<Node<T>>>, value: &T, size: &mut usize) -> Option<Box<Node<T>>> {
        match node {
            None => None,
            Some(mut n) => match value.cmp(&n.value) {
                Ordering::Less => {
                    n.left = Self::remove_node(n.left.take(), value, size);
                    Some(Self::rebalance(n))
                }
                Ordering::Greater => {
                    n.right = Self::remove_node(n.right.take(), value, size);
                    Some(Self::rebalance(n))
                }
                Ordering::Equal => {
                    *size -= 1;
                    match (n.left.take(), n.right.take()) {
                        (None, None) => None,
                        (Some(left), None) => Some(left),
                        (None, Some(right)) => Some(right),
                        (Some(left), Some(right)) => {
                            *size += 1;
                            let (new_right, successor_value) = Self::remove_min_node(right, size);
                            let new_node = Box::new(Node {
                                value: successor_value,
                                left: Some(left),
                                right: new_right,
                                height: 1,
                            });
                            Some(Self::rebalance(new_node))
                        }
                    }
                }
            },
        }
    }

    fn remove_min_node(node: Box<Node<T>>, size: &mut usize) -> (Option<Box<Node<T>>>, T) {
        let mut n = node;
        if n.left.is_none() {
            *size -= 1;
            (n.right.take(), n.value)
        } else {
            let (new_left, min_value) = Self::remove_min_node(n.left.take().unwrap(), size);
            n.left = new_left;
            (Some(Self::rebalance(n)), min_value)
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

    fn check_balance(node: &Option<Box<Node<T>>>) -> bool {
        match node {
            None => true,
            Some(n) => {
                let balance = Self::node_height(&n.left) as i32 - Self::node_height(&n.right) as i32;
                balance.abs() <= 1 && Self::check_balance(&n.left) && Self::check_balance(&n.right)
            }
        }
    }
}

impl<T: Ord> Default for AVLTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord + Clone> Clone for AVLTree<T> {
    fn clone(&self) -> Self {
        let mut new_tree = AVLTree::new();
        for value in self.pre_order() {
            new_tree.insert(value.clone());
        }
        new_tree
    }
}

impl<T: Ord> IntoIterator for AVLTree<T> {
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

    fn verify_bst_property<T: Ord>(tree: &AVLTree<T>) -> bool {
        is_sorted(&tree.in_order())
    }

    fn verify_balance<T: Ord>(node: &Option<Box<Node<T>>>) -> bool {
        match node {
            None => true,
            Some(n) => {
                let balance = AVLTree::<T>::node_height(&n.left) as i32
                    - AVLTree::<T>::node_height(&n.right) as i32;
                balance.abs() <= 1 && verify_balance(&n.left) && verify_balance(&n.right)
            }
        }
    }

    fn verify_avl_property<T: Ord>(tree: &AVLTree<T>) -> bool {
        verify_bst_property(tree) && verify_balance(&tree.root)
    }

    #[test]
    fn new_tree_is_empty() {
        let tree: AVLTree<i32> = AVLTree::new();
        assert_eq!(tree.size(), 0);
        assert!(tree.is_empty());
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn min_on_empty_returns_none() {
        let tree: AVLTree<i32> = AVLTree::new();
        assert!(tree.min().is_none());
    }

    #[test]
    fn max_on_empty_returns_none() {
        let tree: AVLTree<i32> = AVLTree::new();
        assert!(tree.max().is_none());
    }

    #[test]
    fn insert_single_element() {
        let mut tree = AVLTree::new();
        tree.insert(42);
        assert_eq!(tree.size(), 1);
        assert_eq!(tree.height(), 1);
        assert!(!tree.is_empty());
        assert!(tree.contains(&42));
    }

    #[test]
    fn insert_multiple_elements() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        assert_eq!(tree.size(), 3);
        assert!(tree.contains(&50));
        assert!(tree.contains(&30));
        assert!(tree.contains(&70));
    }

    #[test]
    fn insert_duplicate_is_noop() {
        let mut tree = AVLTree::new();
        tree.insert(42);
        tree.insert(42);
        assert_eq!(tree.size(), 1);
    }

    #[test]
    fn insert_maintains_bst_property() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        tree.insert(40);
        assert!(verify_bst_property(&tree));
    }

    #[test]
    fn insert_maintains_balance() {
        let mut tree = AVLTree::new();
        for i in 1..=10 {
            tree.insert(i);
            assert!(verify_avl_property(&tree));
        }
    }

    #[test]
    fn right_rotation_ll_case() {
        let mut tree = AVLTree::new();
        tree.insert(30);
        tree.insert(20);
        tree.insert(10);
        assert!(verify_avl_property(&tree));
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn height_correct_after_right_rotation() {
        let mut tree = AVLTree::new();
        tree.insert(30);
        tree.insert(20);
        assert_eq!(tree.height(), 2);
        tree.insert(10);
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn left_rotation_rr_case() {
        let mut tree = AVLTree::new();
        tree.insert(10);
        tree.insert(20);
        tree.insert(30);
        assert!(verify_avl_property(&tree));
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn height_correct_after_left_rotation() {
        let mut tree = AVLTree::new();
        tree.insert(10);
        tree.insert(20);
        assert_eq!(tree.height(), 2);
        tree.insert(30);
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn left_right_rotation_lr_case() {
        let mut tree = AVLTree::new();
        tree.insert(30);
        tree.insert(10);
        tree.insert(20);
        assert!(verify_avl_property(&tree));
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn height_correct_after_left_right_rotation() {
        let mut tree = AVLTree::new();
        tree.insert(30);
        tree.insert(10);
        tree.insert(20);
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn right_left_rotation_rl_case() {
        let mut tree = AVLTree::new();
        tree.insert(10);
        tree.insert(30);
        tree.insert(20);
        assert!(verify_avl_property(&tree));
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn height_correct_after_right_left_rotation() {
        let mut tree = AVLTree::new();
        tree.insert(10);
        tree.insert(30);
        tree.insert(20);
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn contains_returns_true_for_existing() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        assert!(tree.contains(&50));
        assert!(tree.contains(&30));
        assert!(tree.contains(&70));
    }

    #[test]
    fn contains_returns_false_for_nonexistent() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        assert!(!tree.contains(&30));
        assert!(!tree.contains(&70));
    }

    #[test]
    fn contains_after_insert() {
        let mut tree = AVLTree::new();
        assert!(!tree.contains(&42));
        tree.insert(42);
        assert!(tree.contains(&42));
    }

    #[test]
    fn contains_after_remove() {
        let mut tree = AVLTree::new();
        tree.insert(42);
        assert!(tree.contains(&42));
        tree.remove(&42);
        assert!(!tree.contains(&42));
    }

    #[test]
    fn remove_leaf_node() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.remove(&30);
        assert!(!tree.contains(&30));
        assert!(tree.contains(&50));
        assert!(tree.contains(&70));
        assert_eq!(tree.size(), 2);
        assert!(verify_avl_property(&tree));
    }

    #[test]
    fn remove_node_with_left_child() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        tree.remove(&30);
        assert!(!tree.contains(&30));
        assert!(tree.contains(&50));
        assert!(tree.contains(&20));
        assert!(verify_avl_property(&tree));
    }

    #[test]
    fn remove_node_with_right_child() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(40);
        tree.remove(&30);
        assert!(!tree.contains(&30));
        assert!(tree.contains(&50));
        assert!(tree.contains(&40));
        assert!(verify_avl_property(&tree));
    }

    #[test]
    fn remove_node_with_two_children() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        tree.insert(40);
        tree.remove(&30);
        assert!(!tree.contains(&30));
        assert!(tree.contains(&50));
        assert!(tree.contains(&70));
        assert!(tree.contains(&20));
        assert!(tree.contains(&40));
        assert!(verify_avl_property(&tree));
    }

    #[test]
    fn remove_root_node() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.remove(&50);
        assert!(!tree.contains(&50));
        assert!(tree.contains(&30));
        assert!(tree.contains(&70));
        assert!(verify_avl_property(&tree));
    }

    #[test]
    fn remove_nonexistent_is_noop() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.remove(&100);
        assert_eq!(tree.size(), 1);
        assert!(tree.contains(&50));
    }

    #[test]
    fn size_decrements_after_remove() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        assert_eq!(tree.size(), 3);
        tree.remove(&30);
        assert_eq!(tree.size(), 2);
    }

    #[test]
    fn remove_triggers_rebalancing() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(60);
        tree.insert(80);
        tree.remove(&30);
        assert!(verify_avl_property(&tree));
    }

    #[test]
    fn min_returns_smallest() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        assert_eq!(tree.min(), Some(&20));
    }

    #[test]
    fn max_returns_largest() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(80);
        assert_eq!(tree.max(), Some(&80));
    }

    #[test]
    fn min_max_after_insert() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        assert_eq!(tree.min(), Some(&50));
        assert_eq!(tree.max(), Some(&50));
        tree.insert(30);
        assert_eq!(tree.min(), Some(&30));
        tree.insert(70);
        assert_eq!(tree.max(), Some(&70));
    }

    #[test]
    fn min_max_after_remove() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.remove(&30);
        assert_eq!(tree.min(), Some(&50));
        tree.remove(&70);
        assert_eq!(tree.max(), Some(&50));
    }

    #[test]
    fn height_of_empty_tree() {
        let tree: AVLTree<i32> = AVLTree::new();
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn height_of_single_node() {
        let mut tree = AVLTree::new();
        tree.insert(42);
        assert_eq!(tree.height(), 1);
    }

    #[test]
    fn height_updates_after_insert() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        assert_eq!(tree.height(), 1);
        tree.insert(30);
        assert_eq!(tree.height(), 2);
        tree.insert(70);
        assert_eq!(tree.height(), 2);
        tree.insert(20);
        assert_eq!(tree.height(), 3);
    }

    #[test]
    fn height_updates_after_remove() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        tree.remove(&20);
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn height_is_logarithmic() {
        let mut tree = AVLTree::new();
        for i in 1..=1000 {
            tree.insert(i);
        }
        let max_height = (1000_f64.log2() * 1.45).ceil() as usize;
        assert!(tree.height() <= max_height);
    }

    #[test]
    fn in_order_yields_sorted() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        tree.insert(40);
        assert_eq!(tree.in_order(), vec![&20, &30, &40, &50, &70]);
    }

    #[test]
    fn pre_order_yields_correct_sequence() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        let pre = tree.pre_order();
        assert_eq!(pre.len(), 3);
        assert!(verify_bst_property(&tree));
    }

    #[test]
    fn post_order_yields_correct_sequence() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        let post = tree.post_order();
        assert_eq!(post.len(), 3);
        assert!(verify_bst_property(&tree));
    }

    #[test]
    fn traversals_on_empty_tree() {
        let tree: AVLTree<i32> = AVLTree::new();
        assert!(tree.in_order().is_empty());
        assert!(tree.pre_order().is_empty());
        assert!(tree.post_order().is_empty());
    }

    #[test]
    fn clear_makes_tree_empty() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.clear();
        assert!(tree.is_empty());
        assert_eq!(tree.size(), 0);
        assert_eq!(tree.height(), 0);
    }

    #[test]
    fn clear_on_empty_is_noop() {
        let mut tree: AVLTree<i32> = AVLTree::new();
        tree.clear();
        assert!(tree.is_empty());
    }

    #[test]
    fn avl_property_holds_after_operations() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        tree.insert(40);
        assert!(verify_avl_property(&tree));
        tree.remove(&30);
        assert!(verify_avl_property(&tree));
        tree.insert(35);
        assert!(verify_avl_property(&tree));
    }

    #[test]
    fn sorted_insert_produces_balanced_tree() {
        let mut tree = AVLTree::new();
        for i in 1..=10 {
            tree.insert(i);
        }
        assert!(verify_avl_property(&tree));
        assert!(tree.height() <= 4);
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        let clone = tree.clone();
        assert_eq!(clone.size(), 3);
        assert!(clone.contains(&50));
        assert!(clone.contains(&30));
        assert!(clone.contains(&70));
    }

    #[test]
    fn insert_to_original_doesnt_affect_clone() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        let clone = tree.clone();
        tree.insert(30);
        assert_eq!(tree.size(), 2);
        assert_eq!(clone.size(), 1);
        assert!(!clone.contains(&30));
    }

    #[test]
    fn remove_from_original_doesnt_affect_clone() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        let clone = tree.clone();
        tree.remove(&30);
        assert!(!tree.contains(&30));
        assert!(clone.contains(&30));
    }

    #[test]
    fn single_element_tree() {
        let mut tree = AVLTree::new();
        tree.insert(42);
        assert_eq!(tree.min(), Some(&42));
        assert_eq!(tree.max(), Some(&42));
        assert!(tree.contains(&42));
        tree.remove(&42);
        assert!(tree.is_empty());
    }

    #[test]
    fn reverse_sorted_insert_produces_balanced_tree() {
        let mut tree = AVLTree::new();
        for i in (1..=10).rev() {
            tree.insert(i);
        }
        assert!(verify_avl_property(&tree));
        assert!(tree.height() <= 4);
    }

    #[test]
    fn large_number_of_elements() {
        let mut tree = AVLTree::new();
        for i in 1..=1000 {
            tree.insert(i);
        }
        assert_eq!(tree.size(), 1000);
        assert_eq!(tree.min(), Some(&1));
        assert_eq!(tree.max(), Some(&1000));
        let max_height = (1000_f64.log2() * 1.45).ceil() as usize;
        assert!(tree.height() <= max_height);
    }

    #[test]
    fn negative_numbers() {
        let mut tree = AVLTree::new();
        tree.insert(-10);
        tree.insert(0);
        tree.insert(10);
        tree.insert(-20);
        assert_eq!(tree.min(), Some(&-20));
        assert_eq!(tree.max(), Some(&10));
        assert!(tree.contains(&-10));
        assert!(verify_avl_property(&tree));
    }

    #[test]
    fn remove_all_elements_one_by_one() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        tree.insert(40);
        tree.remove(&20);
        assert!(verify_avl_property(&tree));
        tree.remove(&40);
        assert!(verify_avl_property(&tree));
        tree.remove(&30);
        assert!(verify_avl_property(&tree));
        tree.remove(&70);
        assert!(verify_avl_property(&tree));
        tree.remove(&50);
        assert!(tree.is_empty());
    }

    #[test]
    fn alternating_insert_remove_maintains_balance() {
        let mut tree = AVLTree::new();
        for i in 1..=20 {
            tree.insert(i);
            if i % 3 == 0 {
                tree.remove(&(i - 2));
            }
            assert!(verify_avl_property(&tree));
        }
    }

    #[test]
    fn default_creates_empty_tree() {
        let tree: AVLTree<i32> = AVLTree::default();
        assert!(tree.is_empty());
    }

    #[test]
    fn into_iter_yields_sorted() {
        let mut tree = AVLTree::new();
        tree.insert(50);
        tree.insert(30);
        tree.insert(70);
        tree.insert(20);
        tree.insert(40);
        let result: Vec<i32> = tree.into_iter().collect();
        assert_eq!(result, vec![20, 30, 40, 50, 70]);
    }

    #[test]
    fn works_with_strings() {
        let mut tree = AVLTree::new();
        tree.insert("banana");
        tree.insert("apple");
        tree.insert("cherry");
        assert_eq!(tree.min(), Some(&"apple"));
        assert_eq!(tree.max(), Some(&"cherry"));
        assert_eq!(tree.in_order(), vec![&"apple", &"banana", &"cherry"]);
    }

    #[test]
    fn into_iter_with_owned_strings() {
        let mut tree = AVLTree::new();
        tree.insert(String::from("banana"));
        tree.insert(String::from("apple"));
        tree.insert(String::from("cherry"));
        let result: Vec<String> = tree.into_iter().collect();
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }
}
