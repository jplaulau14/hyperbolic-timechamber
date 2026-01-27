pub struct StaticArray<T, const N: usize> {
    data: [T; N],
}

impl<T: Default, const N: usize> StaticArray<T, N> {
    pub fn new() -> Self {
        Self {
            data: core::array::from_fn(|_| T::default()),
        }
    }
}

impl<T, const N: usize> StaticArray<T, N> {
    pub fn at(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    pub fn at_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)
    }

    pub fn front(&self) -> Option<&T> {
        self.data.first()
    }

    pub fn back(&self) -> Option<&T> {
        self.data.last()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn size(&self) -> usize {
        N
    }

    pub fn is_empty(&self) -> bool {
        N == 0
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<T: Clone, const N: usize> StaticArray<T, N> {
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T, const N: usize> core::ops::Index<usize> for StaticArray<T, N> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T, const N: usize> core::ops::IndexMut<usize> for StaticArray<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.data[index]
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a StaticArray<T, N> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_and_empty() {
        let arr: StaticArray<i32, 5> = StaticArray::new();
        assert_eq!(arr.size(), 5);
        assert!(!arr.is_empty());

        let empty: StaticArray<i32, 0> = StaticArray::new();
        assert_eq!(empty.size(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn fill_and_access() {
        let mut arr: StaticArray<i32, 4> = StaticArray::new();
        arr.fill(42);
        for i in 0..arr.size() {
            assert_eq!(arr[i], 42);
        }
    }

    #[test]
    fn index_operator() {
        let mut arr: StaticArray<i32, 3> = StaticArray::new();
        arr[0] = 10;
        arr[1] = 20;
        arr[2] = 30;
        assert_eq!(arr[0], 10);
        assert_eq!(arr[1], 20);
        assert_eq!(arr[2], 30);
    }

    #[test]
    fn at_valid_index() {
        let mut arr: StaticArray<i32, 3> = StaticArray::new();
        *arr.at_mut(0).unwrap() = 100;
        *arr.at_mut(1).unwrap() = 200;
        *arr.at_mut(2).unwrap() = 300;
        assert_eq!(*arr.at(0).unwrap(), 100);
        assert_eq!(*arr.at(1).unwrap(), 200);
        assert_eq!(*arr.at(2).unwrap(), 300);
    }

    #[test]
    fn at_out_of_range() {
        let arr: StaticArray<i32, 3> = StaticArray::new();
        assert!(arr.at(3).is_none());
        assert!(arr.at(100).is_none());
    }

    #[test]
    fn at_on_zero_size() {
        let arr: StaticArray<i32, 0> = StaticArray::new();
        assert!(arr.at(0).is_none());
    }

    #[test]
    fn front_and_back() {
        let mut arr: StaticArray<i32, 4> = StaticArray::new();
        arr.fill(0);
        arr[0] = 1;
        arr[3] = 99;
        assert_eq!(*arr.front().unwrap(), 1);
        assert_eq!(*arr.back().unwrap(), 99);
    }

    #[test]
    fn data_slice() {
        let mut arr: StaticArray<i32, 3> = StaticArray::new();
        arr.fill(7);
        let d = arr.data();
        assert_eq!(d[0], 7);
        assert_eq!(d[1], 7);
        assert_eq!(d[2], 7);
    }

    #[test]
    fn data_zero_size() {
        let arr: StaticArray<i32, 0> = StaticArray::new();
        assert!(arr.data().is_empty());
    }

    #[test]
    fn iteration() {
        let mut arr: StaticArray<i32, 5> = StaticArray::new();
        arr.fill(3);
        let sum: i32 = arr.iter().sum();
        assert_eq!(sum, 15);
    }

    #[test]
    fn iterator_next() {
        let mut arr: StaticArray<i32, 4> = StaticArray::new();
        for i in 0..4 {
            arr[i] = (i * 10) as i32;
        }
        let mut it = arr.iter();
        assert_eq!(*it.next().unwrap(), 0);
        assert_eq!(*it.next().unwrap(), 10);
    }

    #[test]
    fn into_iterator() {
        let mut arr: StaticArray<i32, 3> = StaticArray::new();
        arr.fill(5);
        let mut sum = 0;
        for v in &arr {
            sum += v;
        }
        assert_eq!(sum, 15);
    }

    #[test]
    fn non_trivial_type() {
        let mut arr: StaticArray<String, 3> = StaticArray::new();
        arr[0] = "hello".to_string();
        arr[1] = "world".to_string();
        arr[2] = "!".to_string();
        assert_eq!(arr[0], "hello");
        assert_eq!(*arr.at(1).unwrap(), "world");
        assert_eq!(*arr.back().unwrap(), "!");
    }

    #[test]
    fn fill_overwrites() {
        let mut arr: StaticArray<i32, 3> = StaticArray::new();
        arr.fill(1);
        arr.fill(2);
        for i in 0..arr.size() {
            assert_eq!(arr[i], 2);
        }
    }

    #[test]
    fn zero_size_iteration() {
        let arr: StaticArray<i32, 0> = StaticArray::new();
        let mut count = 0;
        for _ in arr.iter() {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    #[test]
    fn single_element() {
        let mut arr: StaticArray<i32, 1> = StaticArray::new();
        arr[0] = 42;
        assert_eq!(*arr.front().unwrap(), 42);
        assert_eq!(*arr.back().unwrap(), 42);
        assert_eq!(arr.size(), 1);
    }

    #[test]
    fn front_back_zero_size() {
        let arr: StaticArray<i32, 0> = StaticArray::new();
        assert!(arr.front().is_none());
        assert!(arr.back().is_none());
    }
}
