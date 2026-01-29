class DynamicArray:
    def __init__(self, size=0):
        if not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer")
        self._size = size
        self._capacity = size
        self._data = [None] * size if size > 0 else []

    def at(self, index):
        if index < 0 or index >= self._size:
            raise IndexError("DynamicArray.at: index out of range")
        return self._data[index]

    def set_at(self, index, value):
        if index < 0 or index >= self._size:
            raise IndexError("DynamicArray.set_at: index out of range")
        self._data[index] = value

    def __getitem__(self, index):
        return self._data[index]

    def __setitem__(self, index, value):
        self._data[index] = value

    def front(self):
        if self._size == 0:
            raise IndexError("DynamicArray.front: array is empty")
        return self._data[0]

    def back(self):
        if self._size == 0:
            raise IndexError("DynamicArray.back: array is empty")
        return self._data[self._size - 1]

    def data(self):
        if self._size == 0:
            return None
        return self._data[:self._size]

    def size(self):
        return self._size

    def capacity(self):
        return self._capacity

    def empty(self):
        return self._size == 0

    def reserve(self, new_cap):
        if new_cap <= self._capacity:
            return
        new_data = [None] * new_cap
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        self._capacity = new_cap

    def push_back(self, value):
        if self._size == self._capacity:
            new_cap = 1 if self._capacity == 0 else self._capacity * 2
            self.reserve(new_cap)
        self._data[self._size] = value
        self._size += 1

    def pop_back(self):
        if self._size == 0:
            raise IndexError("DynamicArray.pop_back: array is empty")
        self._size -= 1

    def clear(self):
        self._size = 0

    def copy(self):
        clone = DynamicArray()
        clone._size = self._size
        clone._capacity = self._capacity
        clone._data = [None] * self._capacity
        for i in range(self._size):
            clone._data[i] = self._data[i]
        return clone

    def __iter__(self):
        for i in range(self._size):
            yield self._data[i]

    def __len__(self):
        return self._size
