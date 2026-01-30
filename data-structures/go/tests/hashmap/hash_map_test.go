package hashmap_test

import (
	"testing"

	"github.com/hyperbolic-timechamber/data-structures-go/src/hashmap"
)

func TestNewMapIsEmpty(t *testing.T) {
	m := hashmap.New[string, int]()
	if m.Size() != 0 {
		t.Fatalf("expected size 0, got %d", m.Size())
	}
	if !m.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestDefaultCapacity(t *testing.T) {
	m := hashmap.New[string, int]()
	if m.Capacity() != 16 {
		t.Fatalf("expected capacity 16, got %d", m.Capacity())
	}
}

func TestInsertSinglePair(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("one", 1)
	if m.Size() != 1 {
		t.Fatalf("expected size 1, got %d", m.Size())
	}
	v, ok := m.Get("one")
	if !ok || v != 1 {
		t.Fatalf("expected 1, got %d", v)
	}
}

func TestInsertMultiplePairs(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("one", 1)
	m.Put("two", 2)
	m.Put("three", 3)
	if m.Size() != 3 {
		t.Fatalf("expected size 3, got %d", m.Size())
	}
}

func TestInsertDuplicateKeyUpdatesValue(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("key", 1)
	m.Put("key", 2)
	if m.Size() != 1 {
		t.Fatalf("expected size 1, got %d", m.Size())
	}
	v, _ := m.Get("key")
	if v != 2 {
		t.Fatalf("expected 2, got %d", v)
	}
}

func TestInsertTriggersRehash(t *testing.T) {
	m := hashmap.New[int, int]()
	initialCap := m.Capacity()
	for i := 0; i < 13; i++ {
		m.Put(i, i*10)
	}
	if m.Capacity() <= initialCap {
		t.Fatalf("expected capacity to grow, got %d", m.Capacity())
	}
}

func TestGetExistingKey(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("key", 42)
	v, ok := m.Get("key")
	if !ok {
		t.Fatal("expected key to exist")
	}
	if v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
}

func TestGetNonExistentKey(t *testing.T) {
	m := hashmap.New[string, int]()
	_, ok := m.Get("missing")
	if ok {
		t.Fatal("expected key not to exist")
	}
}

func TestGetAfterUpdate(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("key", 1)
	m.Put("key", 2)
	v, _ := m.Get("key")
	if v != 2 {
		t.Fatalf("expected 2, got %d", v)
	}
}

func TestRemoveExistingKey(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("key", 42)
	v, ok := m.Remove("key")
	if !ok {
		t.Fatal("expected remove to succeed")
	}
	if v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
}

func TestRemoveNonExistentKey(t *testing.T) {
	m := hashmap.New[string, int]()
	_, ok := m.Remove("missing")
	if ok {
		t.Fatal("expected remove to return false")
	}
}

func TestRemoveDecrementsSize(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("a", 1)
	m.Put("b", 2)
	m.Remove("a")
	if m.Size() != 1 {
		t.Fatalf("expected size 1, got %d", m.Size())
	}
}

func TestGetAfterRemove(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("key", 42)
	m.Remove("key")
	_, ok := m.Get("key")
	if ok {
		t.Fatal("expected key not to exist after remove")
	}
}

func TestContainsExistingKey(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("key", 1)
	if !m.Contains("key") {
		t.Fatal("expected contains to return true")
	}
}

func TestContainsNonExistentKey(t *testing.T) {
	m := hashmap.New[string, int]()
	if m.Contains("missing") {
		t.Fatal("expected contains to return false")
	}
}

func TestContainsAfterRemove(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("key", 1)
	m.Remove("key")
	if m.Contains("key") {
		t.Fatal("expected contains to return false after remove")
	}
}

func TestClearMakesMapEmpty(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("a", 1)
	m.Put("b", 2)
	m.Clear()
	if !m.IsEmpty() {
		t.Fatal("expected empty after clear")
	}
	if m.Size() != 0 {
		t.Fatalf("expected size 0, got %d", m.Size())
	}
}

func TestClearOnEmptyMap(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Clear()
	if !m.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestKeysReturnsAllKeys(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("a", 1)
	m.Put("b", 2)
	m.Put("c", 3)
	keys := m.Keys()
	if len(keys) != 3 {
		t.Fatalf("expected 3 keys, got %d", len(keys))
	}
	keySet := make(map[string]bool)
	for _, k := range keys {
		keySet[k] = true
	}
	if !keySet["a"] || !keySet["b"] || !keySet["c"] {
		t.Fatal("missing expected keys")
	}
}

func TestValuesReturnsAllValues(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("a", 1)
	m.Put("b", 2)
	m.Put("c", 3)
	values := m.Values()
	if len(values) != 3 {
		t.Fatalf("expected 3 values, got %d", len(values))
	}
	sum := 0
	for _, v := range values {
		sum += v
	}
	if sum != 6 {
		t.Fatalf("expected sum 6, got %d", sum)
	}
}

func TestKeysAndValuesConsistent(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("a", 1)
	m.Put("b", 2)
	m.Put("c", 3)
	keys := m.Keys()
	values := m.Values()
	if len(keys) != len(values) {
		t.Fatal("keys and values length mismatch")
	}
}

func TestCollisionInsert(t *testing.T) {
	m := hashmap.New[int, string]()
	for i := 0; i < 20; i++ {
		m.Put(i, "value")
	}
	if m.Size() != 20 {
		t.Fatalf("expected size 20, got %d", m.Size())
	}
}

func TestCollisionGet(t *testing.T) {
	m := hashmap.New[int, int]()
	for i := 0; i < 20; i++ {
		m.Put(i, i*10)
	}
	for i := 0; i < 20; i++ {
		v, ok := m.Get(i)
		if !ok {
			t.Fatalf("key %d not found", i)
		}
		if v != i*10 {
			t.Fatalf("expected %d, got %d", i*10, v)
		}
	}
}

func TestCollisionRemove(t *testing.T) {
	m := hashmap.New[int, int]()
	for i := 0; i < 20; i++ {
		m.Put(i, i*10)
	}
	for i := 0; i < 10; i++ {
		m.Remove(i)
	}
	if m.Size() != 10 {
		t.Fatalf("expected size 10, got %d", m.Size())
	}
	for i := 10; i < 20; i++ {
		v, ok := m.Get(i)
		if !ok {
			t.Fatalf("key %d not found", i)
		}
		if v != i*10 {
			t.Fatalf("expected %d, got %d", i*10, v)
		}
	}
}

func TestRehashGrowsCapacity(t *testing.T) {
	m := hashmap.New[int, int]()
	initialCap := m.Capacity()
	for i := 0; i < 13; i++ {
		m.Put(i, i)
	}
	if m.Capacity() != initialCap*2 {
		t.Fatalf("expected capacity %d, got %d", initialCap*2, m.Capacity())
	}
}

func TestAllEntriesAccessibleAfterRehash(t *testing.T) {
	m := hashmap.New[int, int]()
	for i := 0; i < 50; i++ {
		m.Put(i, i*10)
	}
	for i := 0; i < 50; i++ {
		v, ok := m.Get(i)
		if !ok {
			t.Fatalf("key %d not found after rehash", i)
		}
		if v != i*10 {
			t.Fatalf("expected %d, got %d", i*10, v)
		}
	}
}

func TestSizeUnchangedAfterRehash(t *testing.T) {
	m := hashmap.New[int, int]()
	for i := 0; i < 13; i++ {
		m.Put(i, i)
	}
	if m.Size() != 13 {
		t.Fatalf("expected size 13, got %d", m.Size())
	}
}

func TestCloneCreatesIndependentCopy(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("a", 1)
	m.Put("b", 2)
	clone := m.Clone()
	if clone.Size() != 2 {
		t.Fatalf("expected clone size 2, got %d", clone.Size())
	}
	v, ok := clone.Get("a")
	if !ok || v != 1 {
		t.Fatal("expected clone to contain a=1")
	}
}

func TestModifyOriginalDoesNotAffectClone(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("a", 1)
	m.Put("b", 2)
	clone := m.Clone()
	m.Put("c", 3)
	m.Remove("a")
	if clone.Size() != 2 {
		t.Fatalf("expected clone size 2, got %d", clone.Size())
	}
	if !clone.Contains("a") {
		t.Fatal("expected clone to still contain a")
	}
	if clone.Contains("c") {
		t.Fatal("expected clone not to contain c")
	}
}

func TestWorksWithStringKeys(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("hello", 1)
	m.Put("world", 2)
	v, ok := m.Get("hello")
	if !ok || v != 1 {
		t.Fatal("expected hello=1")
	}
}

func TestWorksWithStringValues(t *testing.T) {
	m := hashmap.New[int, string]()
	m.Put(1, "one")
	m.Put(2, "two")
	v, ok := m.Get(1)
	if !ok || v != "one" {
		t.Fatal("expected 1=one")
	}
}

type Point struct {
	X, Y int
}

func TestWorksWithComplexValueTypes(t *testing.T) {
	m := hashmap.New[string, Point]()
	m.Put("origin", Point{0, 0})
	m.Put("point", Point{3, 4})
	v, ok := m.Get("point")
	if !ok || v.X != 3 || v.Y != 4 {
		t.Fatal("expected point={3,4}")
	}
}

func TestSingleElementMap(t *testing.T) {
	m := hashmap.New[string, int]()
	m.Put("only", 42)
	if m.Size() != 1 {
		t.Fatalf("expected size 1, got %d", m.Size())
	}
	v, _ := m.Get("only")
	if v != 42 {
		t.Fatalf("expected 42, got %d", v)
	}
}

func TestLargeNumberOfElements(t *testing.T) {
	m := hashmap.New[int, int]()
	for i := 0; i < 1000; i++ {
		m.Put(i, i*10)
	}
	if m.Size() != 1000 {
		t.Fatalf("expected size 1000, got %d", m.Size())
	}
	for i := 0; i < 1000; i++ {
		v, ok := m.Get(i)
		if !ok || v != i*10 {
			t.Fatalf("expected %d=%d", i, i*10)
		}
	}
}

func TestManyInsertionsAndRemovals(t *testing.T) {
	m := hashmap.New[int, int]()
	for i := 0; i < 500; i++ {
		m.Put(i, i)
	}
	for i := 0; i < 250; i++ {
		m.Remove(i)
	}
	if m.Size() != 250 {
		t.Fatalf("expected size 250, got %d", m.Size())
	}
	for i := 250; i < 500; i++ {
		if !m.Contains(i) {
			t.Fatalf("expected key %d to exist", i)
		}
	}
	for i := 0; i < 250; i++ {
		if m.Contains(i) {
			t.Fatalf("expected key %d to not exist", i)
		}
	}
}
