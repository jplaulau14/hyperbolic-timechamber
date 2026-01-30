package hashmap

import "fmt"

const (
	defaultCapacity = 16
	loadFactorLimit = 0.75
	fnvOffsetBasis  = 14695981039346656037
	fnvPrime        = 1099511628211
)

type entry[K comparable, V any] struct {
	key   K
	value V
	next  *entry[K, V]
}

type HashMap[K comparable, V any] struct {
	buckets  []*entry[K, V]
	size     int
	capacity int
}

func New[K comparable, V any]() *HashMap[K, V] {
	return &HashMap[K, V]{
		buckets:  make([]*entry[K, V], defaultCapacity),
		capacity: defaultCapacity,
	}
}

// fnvHash implements FNV-1a hash algorithm
func fnvHash(s string) uint64 {
	h := uint64(fnvOffsetBasis)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= fnvPrime
	}
	return h
}

func (m *HashMap[K, V]) hash(key K) int {
	h := fnvHash(fmt.Sprintf("%v", key))
	return int(h % uint64(m.capacity))
}

func (m *HashMap[K, V]) Put(key K, value V) {
	if float64(m.size+1)/float64(m.capacity) > loadFactorLimit {
		m.rehash()
	}
	idx := m.hash(key)
	for e := m.buckets[idx]; e != nil; e = e.next {
		if e.key == key {
			e.value = value
			return
		}
	}
	m.buckets[idx] = &entry[K, V]{key: key, value: value, next: m.buckets[idx]}
	m.size++
}

func (m *HashMap[K, V]) Get(key K) (V, bool) {
	var zero V
	idx := m.hash(key)
	for e := m.buckets[idx]; e != nil; e = e.next {
		if e.key == key {
			return e.value, true
		}
	}
	return zero, false
}

func (m *HashMap[K, V]) Remove(key K) (V, bool) {
	var zero V
	idx := m.hash(key)
	var prev *entry[K, V]
	for e := m.buckets[idx]; e != nil; e = e.next {
		if e.key == key {
			if prev == nil {
				m.buckets[idx] = e.next
			} else {
				prev.next = e.next
			}
			m.size--
			return e.value, true
		}
		prev = e
	}
	return zero, false
}

func (m *HashMap[K, V]) Contains(key K) bool {
	_, found := m.Get(key)
	return found
}

func (m *HashMap[K, V]) Size() int {
	return m.size
}

func (m *HashMap[K, V]) IsEmpty() bool {
	return m.size == 0
}

func (m *HashMap[K, V]) Clear() {
	m.buckets = make([]*entry[K, V], m.capacity)
	m.size = 0
}

func (m *HashMap[K, V]) Keys() []K {
	keys := make([]K, 0, m.size)
	for _, bucket := range m.buckets {
		for e := bucket; e != nil; e = e.next {
			keys = append(keys, e.key)
		}
	}
	return keys
}

func (m *HashMap[K, V]) Values() []V {
	values := make([]V, 0, m.size)
	for _, bucket := range m.buckets {
		for e := bucket; e != nil; e = e.next {
			values = append(values, e.value)
		}
	}
	return values
}

func (m *HashMap[K, V]) Capacity() int {
	return m.capacity
}

// Clone creates a copy of this HashMap.
// Note: This performs a shallow copy of values. If values are mutable
// objects (pointers, slices, maps), modifications to them will be visible
// in both the original and cloned HashMap.
func (m *HashMap[K, V]) Clone() *HashMap[K, V] {
	clone := &HashMap[K, V]{
		buckets:  make([]*entry[K, V], m.capacity),
		size:     m.size,
		capacity: m.capacity,
	}
	for i, bucket := range m.buckets {
		var prev *entry[K, V]
		for e := bucket; e != nil; e = e.next {
			newEntry := &entry[K, V]{key: e.key, value: e.value}
			if prev == nil {
				clone.buckets[i] = newEntry
			} else {
				prev.next = newEntry
			}
			prev = newEntry
		}
	}
	return clone
}

func (m *HashMap[K, V]) rehash() {
	oldBuckets := m.buckets
	m.capacity *= 2
	m.buckets = make([]*entry[K, V], m.capacity)
	m.size = 0
	for _, bucket := range oldBuckets {
		for e := bucket; e != nil; e = e.next {
			m.Put(e.key, e.value)
		}
	}
}
