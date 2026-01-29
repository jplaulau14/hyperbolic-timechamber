pub struct DynamicArray<T> {
    data: *mut T,
    size: usize,
    capacity: usize,
}

impl<T> DynamicArray<T> {
    pub fn new() -> Self {
        Self {
            data: core::ptr::null_mut(),
            size: 0,
            capacity: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        let layout = core::alloc::Layout::array::<T>(capacity).unwrap();
        let data = unsafe { std::alloc::alloc(layout) as *mut T };
        if data.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            data,
            size: 0,
            capacity,
        }
    }

    pub fn from_size_with<F: FnMut() -> T>(size: usize, mut f: F) -> Self {
        let mut arr = Self::with_capacity(size);
        for _ in 0..size {
            arr.push_back(f());
        }
        arr
    }

    pub fn at(&self, index: usize) -> Option<&T> {
        if index >= self.size {
            None
        } else {
            unsafe { Some(&*self.data.add(index)) }
        }
    }

    pub fn at_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.size {
            None
        } else {
            unsafe { Some(&mut *self.data.add(index)) }
        }
    }

    pub fn front(&self) -> Option<&T> {
        self.at(0)
    }

    pub fn back(&self) -> Option<&T> {
        if self.size == 0 {
            None
        } else {
            self.at(self.size - 1)
        }
    }

    pub fn data(&self) -> &[T] {
        if self.size == 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.data, self.size) }
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn reserve(&mut self, new_cap: usize) {
        if new_cap <= self.capacity {
            return;
        }
        let new_layout = core::alloc::Layout::array::<T>(new_cap).unwrap();
        let new_data = unsafe { std::alloc::alloc(new_layout) as *mut T };
        if new_data.is_null() {
            std::alloc::handle_alloc_error(new_layout);
        }
        if !self.data.is_null() {
            unsafe {
                core::ptr::copy_nonoverlapping(self.data, new_data, self.size);
                let old_layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
                std::alloc::dealloc(self.data as *mut u8, old_layout);
            }
        }
        self.data = new_data;
        self.capacity = new_cap;
    }

    pub fn push_back(&mut self, value: T) {
        if self.size == self.capacity {
            let new_cap = if self.capacity == 0 { 1 } else { self.capacity * 2 };
            self.reserve(new_cap);
        }
        unsafe {
            core::ptr::write(self.data.add(self.size), value);
        }
        self.size += 1;
    }

    pub fn pop_back(&mut self) -> Option<T> {
        if self.size == 0 {
            None
        } else {
            self.size -= 1;
            unsafe { Some(core::ptr::read(self.data.add(self.size))) }
        }
    }

    pub fn clear(&mut self) {
        while self.size > 0 {
            self.pop_back();
        }
    }

    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.data().iter()
    }
}

impl<T: Clone> Clone for DynamicArray<T> {
    fn clone(&self) -> Self {
        let mut new_arr = Self::with_capacity(self.capacity);
        for i in 0..self.size {
            new_arr.push_back(unsafe { (*self.data.add(i)).clone() });
        }
        new_arr
    }
}

impl<T> Drop for DynamicArray<T> {
    fn drop(&mut self) {
        self.clear();
        if !self.data.is_null() {
            let layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
    }
}

impl<T> core::ops::Index<usize> for DynamicArray<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        self.at(index).expect("index out of bounds")
    }
}

impl<T> core::ops::IndexMut<usize> for DynamicArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        self.at_mut(index).expect("index out of bounds")
    }
}

impl<'a, T> IntoIterator for &'a DynamicArray<T> {
    type Item = &'a T;
    type IntoIter = core::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: Default> Default for DynamicArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Send> Send for DynamicArray<T> {}
unsafe impl<T: Sync> Sync for DynamicArray<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_construction() {
        let arr: DynamicArray<i32> = DynamicArray::new();
        assert_eq!(arr.size(), 0);
        assert_eq!(arr.capacity(), 0);
        assert!(arr.is_empty());
    }

    #[test]
    fn sized_construction() {
        let arr: DynamicArray<i32> = DynamicArray::from_size_with(5, || 0);
        assert_eq!(arr.size(), 5);
        assert_eq!(arr.capacity(), 5);
        assert!(!arr.is_empty());
        for i in 0..arr.size() {
            assert_eq!(arr[i], 0);
        }
    }

    #[test]
    fn push_back_single() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(42);
        assert_eq!(arr.size(), 1);
        assert_eq!(arr[0], 42);
    }

    #[test]
    fn push_back_multiple() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        arr.push_back(2);
        arr.push_back(3);
        assert_eq!(arr.size(), 3);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);
    }

    #[test]
    fn push_back_triggers_growth() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        assert_eq!(arr.capacity(), 1);
        arr.push_back(2);
        assert_eq!(arr.capacity(), 2);
        arr.push_back(3);
        assert_eq!(arr.capacity(), 4);
        arr.push_back(4);
        arr.push_back(5);
        assert_eq!(arr.capacity(), 8);
        assert_eq!(arr.size(), 5);
    }

    #[test]
    fn pop_back() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(10);
        arr.push_back(20);
        arr.push_back(30);
        assert_eq!(arr.pop_back(), Some(30));
        assert_eq!(arr.size(), 2);
        assert_eq!(*arr.back().unwrap(), 20);
        assert_eq!(arr.pop_back(), Some(20));
        assert_eq!(arr.size(), 1);
        assert_eq!(*arr.back().unwrap(), 10);
    }

    #[test]
    fn at_valid_index() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(100);
        arr.push_back(200);
        arr.push_back(300);
        assert_eq!(*arr.at(0).unwrap(), 100);
        assert_eq!(*arr.at(1).unwrap(), 200);
        assert_eq!(*arr.at(2).unwrap(), 300);
        *arr.at_mut(1).unwrap() = 999;
        assert_eq!(*arr.at(1).unwrap(), 999);
    }

    #[test]
    fn at_out_of_range() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        assert!(arr.at(1).is_none());
        assert!(arr.at(100).is_none());
        let empty: DynamicArray<i32> = DynamicArray::new();
        assert!(empty.at(0).is_none());
    }

    #[test]
    fn index_operator() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(10);
        arr.push_back(20);
        assert_eq!(arr[0], 10);
        assert_eq!(arr[1], 20);
        arr[0] = 99;
        assert_eq!(arr[0], 99);
    }

    #[test]
    fn front_and_back() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        arr.push_back(2);
        arr.push_back(3);
        assert_eq!(*arr.front().unwrap(), 1);
        assert_eq!(*arr.back().unwrap(), 3);
    }

    #[test]
    fn reserve_increases_capacity() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.reserve(10);
        assert!(arr.capacity() >= 10);
        assert_eq!(arr.size(), 0);
    }

    #[test]
    fn reserve_preserves_elements() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        arr.push_back(2);
        arr.push_back(3);
        arr.reserve(100);
        assert!(arr.capacity() >= 100);
        assert_eq!(arr.size(), 3);
        assert_eq!(arr[0], 1);
        assert_eq!(arr[1], 2);
        assert_eq!(arr[2], 3);
    }

    #[test]
    fn reserve_smaller_is_noop() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.reserve(10);
        let cap = arr.capacity();
        arr.reserve(5);
        assert_eq!(arr.capacity(), cap);
    }

    #[test]
    fn clear_resets_size_not_capacity() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        arr.push_back(2);
        arr.push_back(3);
        let cap = arr.capacity();
        arr.clear();
        assert_eq!(arr.size(), 0);
        assert!(arr.is_empty());
        assert_eq!(arr.capacity(), cap);
    }

    #[test]
    fn clone_array() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        arr.push_back(2);
        arr.push_back(3);

        let clone = arr.clone();
        assert_eq!(clone.size(), 3);
        assert_eq!(clone[0], 1);
        assert_eq!(clone[1], 2);
        assert_eq!(clone[2], 3);

        arr[0] = 999;
        assert_eq!(clone[0], 1);
    }

    #[test]
    fn iteration() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        arr.push_back(2);
        arr.push_back(3);
        arr.push_back(4);
        let sum: i32 = arr.iter().sum();
        assert_eq!(sum, 10);
    }

    #[test]
    fn into_iterator() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        arr.push_back(2);
        arr.push_back(3);
        let mut sum = 0;
        for v in &arr {
            sum += v;
        }
        assert_eq!(sum, 6);
    }

    #[test]
    fn non_trivial_type() {
        let mut arr: DynamicArray<String> = DynamicArray::new();
        arr.push_back("hello".to_string());
        arr.push_back("world".to_string());
        assert_eq!(arr.size(), 2);
        assert_eq!(arr[0], "hello");
        assert_eq!(arr[1], "world");
        assert_eq!(*arr.front().unwrap(), "hello");
        assert_eq!(*arr.back().unwrap(), "world");
    }

    #[test]
    fn data_slice() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        arr.push_back(1);
        arr.push_back(2);
        let d = arr.data();
        assert_eq!(d[0], 1);
        assert_eq!(d[1], 2);
    }

    #[test]
    fn empty_data_slice() {
        let arr: DynamicArray<i32> = DynamicArray::new();
        assert!(arr.data().is_empty());
    }

    #[test]
    fn front_back_empty() {
        let arr: DynamicArray<i32> = DynamicArray::new();
        assert!(arr.front().is_none());
        assert!(arr.back().is_none());
    }

    #[test]
    fn pop_back_empty() {
        let mut arr: DynamicArray<i32> = DynamicArray::new();
        assert!(arr.pop_back().is_none());
    }
}
