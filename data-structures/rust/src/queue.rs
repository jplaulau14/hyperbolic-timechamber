pub struct Queue<T> {
    data: *mut T,
    head: usize,
    tail: usize,
    size: usize,
    capacity: usize,
}

impl<T> Queue<T> {
    pub fn new() -> Self {
        Self {
            data: core::ptr::null_mut(),
            head: 0,
            tail: 0,
            size: 0,
            capacity: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        let layout = core::alloc::Layout::array::<T>(capacity).unwrap();
        // SAFETY: Layout is valid (non-zero capacity checked above)
        let data = unsafe { std::alloc::alloc(layout) as *mut T };
        if data.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            data,
            head: 0,
            tail: 0,
            size: 0,
            capacity,
        }
    }

    pub fn enqueue(&mut self, value: T) {
        if self.size == self.capacity {
            self.grow();
        }
        // SAFETY: grow() ensures capacity > size, so tail is in bounds.
        // data is non-null after grow(). The slot at tail is uninitialized.
        unsafe {
            core::ptr::write(self.data.add(self.tail), value);
        }
        self.tail = (self.tail + 1) % self.capacity;
        self.size += 1;
    }

    pub fn dequeue(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        // SAFETY: size > 0 implies data is non-null, head is in bounds,
        // and data[head] was initialized via ptr::write in enqueue.
        let value = unsafe { core::ptr::read(self.data.add(self.head)) };
        self.head = (self.head + 1) % self.capacity;
        self.size -= 1;
        Some(value)
    }

    pub fn front(&self) -> Option<&T> {
        if self.size == 0 {
            None
        } else {
            // SAFETY: size > 0 implies data is non-null, head is in bounds,
            // and data[head] was initialized via ptr::write in enqueue.
            unsafe { Some(&*self.data.add(self.head)) }
        }
    }

    pub fn back(&self) -> Option<&T> {
        if self.size == 0 {
            None
        } else {
            let idx = if self.tail == 0 { self.capacity - 1 } else { self.tail - 1 };
            // SAFETY: size > 0 implies data is non-null and idx is valid.
            // The element at idx was the last enqueued, so it is initialized.
            unsafe { Some(&*self.data.add(idx)) }
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn clear(&mut self) {
        while self.dequeue().is_some() {}
    }

    fn grow(&mut self) {
        let new_cap = if self.capacity == 0 { 1 } else { self.capacity * 2 };
        let new_layout = core::alloc::Layout::array::<T>(new_cap).unwrap();
        // SAFETY: Layout is valid (new_cap > 0)
        let new_data = unsafe { std::alloc::alloc(new_layout) as *mut T };
        if new_data.is_null() {
            std::alloc::handle_alloc_error(new_layout);
        }

        // SAFETY: We copy initialized elements from old buffer to new buffer.
        // Source indices are valid (in bounds of old allocation).
        // Dest indices are valid (i < size <= new_cap).
        // Elements are moved (not cloned), so we deallocate old buffer without dropping.
        for i in 0..self.size {
            let src_idx = (self.head + i) % self.capacity;
            unsafe {
                core::ptr::copy_nonoverlapping(self.data.add(src_idx), new_data.add(i), 1);
            }
        }

        if !self.data.is_null() {
            let old_layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
            // SAFETY: data was allocated with this layout, elements were moved out.
            unsafe { std::alloc::dealloc(self.data as *mut u8, old_layout) };
        }

        self.data = new_data;
        self.head = 0;
        self.tail = self.size;
        self.capacity = new_cap;
    }
}

impl<T: Clone> Clone for Queue<T> {
    fn clone(&self) -> Self {
        let mut new_queue = Self::with_capacity(self.capacity);
        for i in 0..self.size {
            let idx = (self.head + i) % self.capacity;
            // SAFETY: i < size implies idx is in bounds of initialized elements.
            let value = unsafe { (*self.data.add(idx)).clone() };
            new_queue.enqueue(value);
        }
        new_queue
    }
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        self.clear();
        if !self.data.is_null() {
            let layout = core::alloc::Layout::array::<T>(self.capacity).unwrap();
            // SAFETY: data was allocated with this layout, clear() dropped all elements.
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
    }
}

// SAFETY: Queue<T> owns its data and can be sent between threads if T can be.
// The raw pointer `data` points to memory exclusively owned by this Queue.
unsafe impl<T: Send> Send for Queue<T> {}

// SAFETY: &Queue<T> only provides read access (front, back, size, is_empty).
// Mutation requires &mut Queue<T>, which Rust's borrow checker enforces.
unsafe impl<T: Sync> Sync for Queue<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_queue_is_empty() {
        let q: Queue<i32> = Queue::new();
        assert_eq!(q.size(), 0);
        assert!(q.is_empty());
    }

    #[test]
    fn front_on_empty_returns_none() {
        let q: Queue<i32> = Queue::new();
        assert!(q.front().is_none());
    }

    #[test]
    fn back_on_empty_returns_none() {
        let q: Queue<i32> = Queue::new();
        assert!(q.back().is_none());
    }

    #[test]
    fn dequeue_on_empty_returns_none() {
        let mut q: Queue<i32> = Queue::new();
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn enqueue_single_element() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(42);
        assert_eq!(q.size(), 1);
        assert!(!q.is_empty());
    }

    #[test]
    fn enqueue_multiple_elements() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        assert_eq!(q.size(), 3);
    }

    #[test]
    fn front_and_back_after_enqueues() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(10);
        q.enqueue(20);
        q.enqueue(30);
        assert_eq!(*q.front().unwrap(), 10);
        assert_eq!(*q.back().unwrap(), 30);
    }

    #[test]
    fn dequeue_returns_front_element() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(100);
        q.enqueue(200);
        assert_eq!(q.dequeue(), Some(100));
    }

    #[test]
    fn dequeue_decrements_size() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        assert_eq!(q.size(), 2);
        q.dequeue();
        assert_eq!(q.size(), 1);
    }

    #[test]
    fn dequeue_all_until_empty() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        q.dequeue();
        q.dequeue();
        q.dequeue();
        assert!(q.is_empty());
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn fifo_order() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        assert_eq!(q.dequeue(), Some(1));
        assert_eq!(q.dequeue(), Some(2));
        assert_eq!(q.dequeue(), Some(3));
    }

    #[test]
    fn front_does_not_remove() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(42);
        assert_eq!(*q.front().unwrap(), 42);
        assert_eq!(*q.front().unwrap(), 42);
        assert_eq!(q.size(), 1);
    }

    #[test]
    fn back_does_not_remove() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(42);
        assert_eq!(*q.back().unwrap(), 42);
        assert_eq!(*q.back().unwrap(), 42);
        assert_eq!(q.size(), 1);
    }

    #[test]
    fn front_and_back_same_for_single_element() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(99);
        assert_eq!(q.front(), q.back());
    }

    #[test]
    fn circular_wrap_around() {
        let mut q: Queue<i32> = Queue::with_capacity(4);
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        q.enqueue(4);
        q.dequeue();
        q.dequeue();
        q.enqueue(5);
        q.enqueue(6);
        assert_eq!(q.dequeue(), Some(3));
        assert_eq!(q.dequeue(), Some(4));
        assert_eq!(q.dequeue(), Some(5));
        assert_eq!(q.dequeue(), Some(6));
    }

    #[test]
    fn fill_dequeue_some_enqueue_more() {
        let mut q: Queue<i32> = Queue::with_capacity(3);
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        q.dequeue();
        q.dequeue();
        q.enqueue(4);
        q.enqueue(5);
        assert_eq!(*q.front().unwrap(), 3);
        assert_eq!(*q.back().unwrap(), 5);
        assert_eq!(q.size(), 3);
    }

    #[test]
    fn growth_preserves_order() {
        let mut q: Queue<i32> = Queue::with_capacity(2);
        q.enqueue(1);
        q.enqueue(2);
        q.dequeue();
        q.enqueue(3);
        q.enqueue(4);
        assert_eq!(q.dequeue(), Some(2));
        assert_eq!(q.dequeue(), Some(3));
        assert_eq!(q.dequeue(), Some(4));
    }

    #[test]
    fn clear_makes_queue_empty() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        q.clear();
        assert!(q.is_empty());
        assert_eq!(q.size(), 0);
    }

    #[test]
    fn clear_on_empty_is_noop() {
        let mut q: Queue<i32> = Queue::new();
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn enqueue_after_clear() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        q.clear();
        q.enqueue(2);
        assert_eq!(q.size(), 1);
        assert_eq!(*q.front().unwrap(), 2);
    }

    #[test]
    fn size_after_enqueues() {
        let mut q: Queue<i32> = Queue::new();
        for i in 0..10 {
            q.enqueue(i);
        }
        assert_eq!(q.size(), 10);
    }

    #[test]
    fn size_after_dequeues() {
        let mut q: Queue<i32> = Queue::new();
        for i in 0..10 {
            q.enqueue(i);
        }
        for _ in 0..5 {
            q.dequeue();
        }
        assert_eq!(q.size(), 5);
    }

    #[test]
    fn is_empty_only_when_size_zero() {
        let mut q: Queue<i32> = Queue::new();
        assert!(q.is_empty());
        q.enqueue(1);
        assert!(!q.is_empty());
        q.dequeue();
        assert!(q.is_empty());
    }

    #[test]
    fn clone_creates_independent_copy() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        q.enqueue(3);
        let clone = q.clone();
        assert_eq!(clone.size(), 3);
        assert_eq!(*clone.front().unwrap(), 1);
        assert_eq!(*clone.back().unwrap(), 3);
    }

    #[test]
    fn enqueue_to_original_does_not_affect_clone() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        let clone = q.clone();
        q.enqueue(2);
        assert_eq!(q.size(), 2);
        assert_eq!(clone.size(), 1);
    }

    #[test]
    fn dequeue_from_original_does_not_affect_clone() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        let clone = q.clone();
        q.dequeue();
        assert_eq!(q.size(), 1);
        assert_eq!(clone.size(), 2);
    }

    #[test]
    fn works_with_strings() {
        let mut q: Queue<String> = Queue::new();
        q.enqueue("hello".to_string());
        q.enqueue("world".to_string());
        assert_eq!(*q.front().unwrap(), "hello");
        assert_eq!(*q.back().unwrap(), "world");
        assert_eq!(q.dequeue(), Some("hello".to_string()));
        assert_eq!(q.dequeue(), Some("world".to_string()));
    }

    #[test]
    fn works_with_custom_types() {
        #[derive(Clone, PartialEq, Debug)]
        struct Point { x: i32, y: i32 }

        let mut q: Queue<Point> = Queue::new();
        q.enqueue(Point { x: 1, y: 2 });
        q.enqueue(Point { x: 3, y: 4 });
        assert_eq!(q.front().unwrap().x, 1);
        assert_eq!(q.back().unwrap().x, 3);
    }

    #[test]
    fn single_element_enqueue_dequeue() {
        let mut q: Queue<i32> = Queue::new();
        q.enqueue(42);
        assert_eq!(q.dequeue(), Some(42));
        assert!(q.is_empty());
    }

    #[test]
    fn alternating_enqueue_dequeue() {
        let mut q: Queue<i32> = Queue::new();
        for i in 0..100 {
            q.enqueue(i);
            assert_eq!(q.dequeue(), Some(i));
        }
        assert!(q.is_empty());
    }

    #[test]
    fn large_number_of_elements() {
        let mut q: Queue<i32> = Queue::new();
        for i in 0..1000 {
            q.enqueue(i);
        }
        assert_eq!(q.size(), 1000);
        for i in 0..1000 {
            assert_eq!(q.dequeue(), Some(i));
        }
        assert!(q.is_empty());
    }

    #[test]
    fn many_wrap_around_cycles() {
        let mut q: Queue<i32> = Queue::with_capacity(4);
        for cycle in 0..100 {
            for i in 0..3 {
                q.enqueue(cycle * 3 + i);
            }
            for i in 0..3 {
                assert_eq!(q.dequeue(), Some(cycle * 3 + i));
            }
        }
        assert!(q.is_empty());
    }

    #[test]
    fn default_trait() {
        let q: Queue<i32> = Queue::default();
        assert!(q.is_empty());
        assert_eq!(q.size(), 0);
    }
}
