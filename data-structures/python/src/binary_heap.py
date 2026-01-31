from typing import TypeVar, Generic, List, Iterator

T = TypeVar('T')


class BinaryHeap(Generic[T]):
    def __init__(self) -> None:
        self._data: List[T] = []

    def push(self, value: T) -> None:
        self._data.append(value)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> T:
        if not self._data:
            raise IndexError("pop from empty heap")
        if len(self._data) == 1:
            return self._data.pop()
        result = self._data[0]
        self._data[0] = self._data.pop()
        self._sift_down(0)
        return result

    def peek(self) -> T:
        if not self._data:
            raise IndexError("peek from empty heap")
        return self._data[0]

    def size(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def clear(self) -> None:
        self._data.clear()

    def copy(self) -> 'BinaryHeap[T]':
        clone: BinaryHeap[T] = BinaryHeap()
        clone._data = self._data.copy()
        return clone

    @staticmethod
    def from_array(arr: List[T]) -> 'BinaryHeap[T]':
        """Build a heap from an array.

        Note: Creates a shallow copy of the input array.
        """
        heap: BinaryHeap[T] = BinaryHeap()
        heap._data = list(arr)
        for i in range(len(heap._data) // 2 - 1, -1, -1):
            heap._sift_down(i)
        return heap

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self._data[index] < self._data[parent]:
                self._data[index], self._data[parent] = self._data[parent], self._data[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        size = len(self._data)
        while True:
            smallest = index
            left = 2 * index + 1
            right = 2 * index + 2
            if left < size and self._data[left] < self._data[smallest]:
                smallest = left
            if right < size and self._data[right] < self._data[smallest]:
                smallest = right
            if smallest == index:
                break
            self._data[index], self._data[smallest] = self._data[smallest], self._data[index]
            index = smallest

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return len(self._data) > 0

    def __repr__(self) -> str:
        return f"BinaryHeap({self._data})"

    def __str__(self) -> str:
        return f"BinaryHeap(size={len(self._data)})"

    def __iter__(self) -> Iterator[T]:
        heap_copy = self.copy()
        while not heap_copy.is_empty():
            yield heap_copy.pop()
