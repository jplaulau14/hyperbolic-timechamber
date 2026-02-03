package avl_test

import (
	"errors"
	"math"
	"sort"
	"testing"

	"github.com/hyperbolic-timechamber/data-structures-go/src/avl"
)

func isSorted(s []int) bool {
	for i := 1; i < len(s); i++ {
		if s[i] < s[i-1] {
			return false
		}
	}
	return true
}

func verifyBSTProperty(t *avl.AVLTree[int]) bool {
	return isSorted(t.InOrder())
}

func verifyAVLBalance(t *avl.AVLTree[int]) bool {
	return checkBalance(t.InOrder(), t)
}

func checkBalance(values []int, tree *avl.AVLTree[int]) bool {
	for _, v := range values {
		if !tree.Contains(v) {
			return false
		}
	}
	h := tree.Height()
	n := tree.Size()
	if n == 0 {
		return h == 0
	}
	maxHeight := int(1.44*math.Log2(float64(n+2)) + 1)
	return h <= maxHeight
}

func TestNewTreeIsEmpty(t *testing.T) {
	tree := avl.New[int]()
	if tree.Size() != 0 {
		t.Fatalf("expected size 0, got %d", tree.Size())
	}
	if !tree.IsEmpty() {
		t.Fatal("expected empty")
	}
	if tree.Height() != 0 {
		t.Fatalf("expected height 0, got %d", tree.Height())
	}
}

func TestMinOnEmptyTree(t *testing.T) {
	tree := avl.New[int]()
	_, err := tree.Min()
	if !errors.Is(err, avl.ErrEmptyTree) {
		t.Fatal("expected ErrEmptyTree")
	}
}

func TestMaxOnEmptyTree(t *testing.T) {
	tree := avl.New[int]()
	_, err := tree.Max()
	if !errors.Is(err, avl.ErrEmptyTree) {
		t.Fatal("expected ErrEmptyTree")
	}
}

func TestInsertSingleElement(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(42)
	if tree.Size() != 1 {
		t.Fatalf("expected size 1, got %d", tree.Size())
	}
	if tree.IsEmpty() {
		t.Fatal("expected not empty")
	}
	if !tree.Contains(42) {
		t.Fatal("expected to contain 42")
	}
	if tree.Height() != 1 {
		t.Fatalf("expected height 1, got %d", tree.Height())
	}
}

func TestInsertMultipleElements(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	if tree.Size() != 3 {
		t.Fatalf("expected size 3, got %d", tree.Size())
	}
	if !tree.Contains(50) || !tree.Contains(30) || !tree.Contains(70) {
		t.Fatal("expected to contain all inserted values")
	}
}

func TestInsertDuplicateIsNoop(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(42)
	tree.Insert(42)
	if tree.Size() != 1 {
		t.Fatalf("expected size 1, got %d", tree.Size())
	}
}

func TestInsertMaintainsBSTProperty(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Insert(40)
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated")
	}
}

func TestInsertMaintainsBalance(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Insert(40)
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated")
	}
}

func TestRightRotationLL(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(30)
	tree.Insert(20)
	tree.Insert(10)
	if tree.Height() != 2 {
		t.Fatalf("expected height 2 after LL rotation, got %d", tree.Height())
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated after LL rotation")
	}
	expected := []int{10, 20, 30}
	result := tree.InOrder()
	for i, v := range expected {
		if result[i] != v {
			t.Fatalf("expected %d at index %d, got %d", v, i, result[i])
		}
	}
}

func TestLeftRotationRR(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(10)
	tree.Insert(20)
	tree.Insert(30)
	if tree.Height() != 2 {
		t.Fatalf("expected height 2 after RR rotation, got %d", tree.Height())
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated after RR rotation")
	}
	expected := []int{10, 20, 30}
	result := tree.InOrder()
	for i, v := range expected {
		if result[i] != v {
			t.Fatalf("expected %d at index %d, got %d", v, i, result[i])
		}
	}
}

func TestLeftRightRotationLR(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(30)
	tree.Insert(10)
	tree.Insert(20)
	if tree.Height() != 2 {
		t.Fatalf("expected height 2 after LR rotation, got %d", tree.Height())
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated after LR rotation")
	}
	expected := []int{10, 20, 30}
	result := tree.InOrder()
	for i, v := range expected {
		if result[i] != v {
			t.Fatalf("expected %d at index %d, got %d", v, i, result[i])
		}
	}
}

func TestRightLeftRotationRL(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(10)
	tree.Insert(30)
	tree.Insert(20)
	if tree.Height() != 2 {
		t.Fatalf("expected height 2 after RL rotation, got %d", tree.Height())
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated after RL rotation")
	}
	expected := []int{10, 20, 30}
	result := tree.InOrder()
	for i, v := range expected {
		if result[i] != v {
			t.Fatalf("expected %d at index %d, got %d", v, i, result[i])
		}
	}
}

func TestContainsReturnsTrueForExisting(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	if !tree.Contains(50) || !tree.Contains(30) || !tree.Contains(70) {
		t.Fatal("expected to contain all inserted values")
	}
}

func TestContainsReturnsFalseForNonexistent(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	if tree.Contains(30) || tree.Contains(70) {
		t.Fatal("expected not to contain 30 or 70")
	}
}

func TestContainsAfterInsert(t *testing.T) {
	tree := avl.New[int]()
	if tree.Contains(42) {
		t.Fatal("expected not to contain 42")
	}
	tree.Insert(42)
	if !tree.Contains(42) {
		t.Fatal("expected to contain 42")
	}
}

func TestContainsAfterRemove(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(42)
	if !tree.Contains(42) {
		t.Fatal("expected to contain 42")
	}
	tree.Remove(42)
	if tree.Contains(42) {
		t.Fatal("expected not to contain 42")
	}
}

func TestRemoveLeafNode(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Remove(30)
	if tree.Contains(30) {
		t.Fatal("expected not to contain 30")
	}
	if !tree.Contains(50) || !tree.Contains(70) {
		t.Fatal("expected to contain 50 and 70")
	}
	if tree.Size() != 2 {
		t.Fatalf("expected size 2, got %d", tree.Size())
	}
}

func TestRemoveNodeWithLeftChild(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Remove(30)
	if tree.Contains(30) {
		t.Fatal("expected not to contain 30")
	}
	if !tree.Contains(50) || !tree.Contains(70) || !tree.Contains(20) {
		t.Fatal("expected to contain 50, 70, 20")
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated")
	}
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated")
	}
}

func TestRemoveNodeWithRightChild(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(40)
	tree.Remove(30)
	if tree.Contains(30) {
		t.Fatal("expected not to contain 30")
	}
	if !tree.Contains(50) || !tree.Contains(70) || !tree.Contains(40) {
		t.Fatal("expected to contain 50, 70, 40")
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated")
	}
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated")
	}
}

func TestRemoveNodeWithTwoChildren(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Insert(40)
	tree.Remove(30)
	if tree.Contains(30) {
		t.Fatal("expected not to contain 30")
	}
	if !tree.Contains(50) || !tree.Contains(70) || !tree.Contains(20) || !tree.Contains(40) {
		t.Fatal("expected to contain 50, 70, 20, 40")
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated")
	}
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated")
	}
}

func TestRemoveRootNode(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Remove(50)
	if tree.Contains(50) {
		t.Fatal("expected not to contain 50")
	}
	if !tree.Contains(30) || !tree.Contains(70) {
		t.Fatal("expected to contain 30 and 70")
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated")
	}
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated")
	}
}

func TestRemoveNonexistentIsNoop(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Remove(100)
	if tree.Size() != 1 {
		t.Fatalf("expected size 1, got %d", tree.Size())
	}
	if !tree.Contains(50) {
		t.Fatal("expected to contain 50")
	}
}

func TestSizeDecrementsAfterRemove(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	if tree.Size() != 3 {
		t.Fatalf("expected size 3, got %d", tree.Size())
	}
	tree.Remove(30)
	if tree.Size() != 2 {
		t.Fatalf("expected size 2, got %d", tree.Size())
	}
}

func TestRemoveTriggersRebalancing(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Insert(40)
	tree.Insert(60)
	tree.Insert(80)
	tree.Insert(10)
	tree.Remove(60)
	tree.Remove(80)
	tree.Remove(70)
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated after rebalancing remove")
	}
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated after rebalancing remove")
	}
}

func TestMinReturnsSmallest(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	v, _ := tree.Min()
	if v != 20 {
		t.Fatalf("expected min 20, got %d", v)
	}
}

func TestMaxReturnsLargest(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(80)
	v, _ := tree.Max()
	if v != 80 {
		t.Fatalf("expected max 80, got %d", v)
	}
}

func TestMinMaxAfterInsert(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	v, _ := tree.Min()
	if v != 50 {
		t.Fatal("expected min 50")
	}
	v, _ = tree.Max()
	if v != 50 {
		t.Fatal("expected max 50")
	}
	tree.Insert(30)
	v, _ = tree.Min()
	if v != 30 {
		t.Fatal("expected min 30")
	}
	tree.Insert(70)
	v, _ = tree.Max()
	if v != 70 {
		t.Fatal("expected max 70")
	}
}

func TestMinMaxAfterRemove(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Remove(30)
	v, _ := tree.Min()
	if v != 50 {
		t.Fatal("expected min 50")
	}
	tree.Remove(70)
	v, _ = tree.Max()
	if v != 50 {
		t.Fatal("expected max 50")
	}
}

func TestHeightOfEmptyTree(t *testing.T) {
	tree := avl.New[int]()
	if tree.Height() != 0 {
		t.Fatalf("expected height 0, got %d", tree.Height())
	}
}

func TestHeightOfSingleNode(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(42)
	if tree.Height() != 1 {
		t.Fatalf("expected height 1, got %d", tree.Height())
	}
}

func TestHeightUpdatesAfterInsert(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	if tree.Height() != 1 {
		t.Fatalf("expected height 1, got %d", tree.Height())
	}
	tree.Insert(30)
	if tree.Height() != 2 {
		t.Fatalf("expected height 2, got %d", tree.Height())
	}
	tree.Insert(70)
	if tree.Height() != 2 {
		t.Fatalf("expected height 2, got %d", tree.Height())
	}
	tree.Insert(20)
	if tree.Height() != 3 {
		t.Fatalf("expected height 3, got %d", tree.Height())
	}
}

func TestHeightUpdatesAfterRemove(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	initialHeight := tree.Height()
	tree.Remove(20)
	if tree.Height() > initialHeight {
		t.Fatal("height should not increase after remove")
	}
}

func TestHeightIsLogN(t *testing.T) {
	tree := avl.New[int]()
	for i := 1; i <= 100; i++ {
		tree.Insert(i)
	}
	h := tree.Height()
	maxExpected := int(1.44*math.Log2(102) + 1)
	if h > maxExpected {
		t.Fatalf("height %d exceeds expected max %d for 100 elements", h, maxExpected)
	}
}

func TestInOrderYieldsSorted(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Insert(40)
	result := tree.InOrder()
	expected := []int{20, 30, 40, 50, 70}
	if len(result) != len(expected) {
		t.Fatal("length mismatch")
	}
	for i := range expected {
		if result[i] != expected[i] {
			t.Fatalf("index %d: expected %d, got %d", i, expected[i], result[i])
		}
	}
}

func TestPreOrderYieldsCorrectSequence(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(20)
	tree.Insert(10)
	tree.Insert(30)
	result := tree.PreOrder()
	expected := []int{20, 10, 30}
	if len(result) != len(expected) {
		t.Fatal("length mismatch")
	}
	for i := range expected {
		if result[i] != expected[i] {
			t.Fatalf("index %d: expected %d, got %d", i, expected[i], result[i])
		}
	}
}

func TestPostOrderYieldsCorrectSequence(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(20)
	tree.Insert(10)
	tree.Insert(30)
	result := tree.PostOrder()
	expected := []int{10, 30, 20}
	if len(result) != len(expected) {
		t.Fatal("length mismatch")
	}
	for i := range expected {
		if result[i] != expected[i] {
			t.Fatalf("index %d: expected %d, got %d", i, expected[i], result[i])
		}
	}
}

func TestTraversalsOnEmptyTree(t *testing.T) {
	tree := avl.New[int]()
	if len(tree.InOrder()) != 0 {
		t.Fatal("expected empty in-order")
	}
	if len(tree.PreOrder()) != 0 {
		t.Fatal("expected empty pre-order")
	}
	if len(tree.PostOrder()) != 0 {
		t.Fatal("expected empty post-order")
	}
}

func TestClearMakesTreeEmpty(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Clear()
	if !tree.IsEmpty() {
		t.Fatal("expected empty")
	}
	if tree.Size() != 0 {
		t.Fatalf("expected size 0, got %d", tree.Size())
	}
	if tree.Height() != 0 {
		t.Fatalf("expected height 0, got %d", tree.Height())
	}
}

func TestClearOnEmptyIsNoop(t *testing.T) {
	tree := avl.New[int]()
	tree.Clear()
	if !tree.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestBSTPropertyHoldsAfterOperations(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Insert(40)
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated")
	}
	tree.Remove(30)
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated after remove")
	}
	tree.Insert(35)
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated after insert")
	}
}

func TestAVLBalanceAfterOperations(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Insert(40)
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated")
	}
	tree.Remove(30)
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated after remove")
	}
	tree.Insert(35)
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated after insert")
	}
}

func TestSortedInsertProducesBalancedTree(t *testing.T) {
	tree := avl.New[int]()
	for i := 1; i <= 15; i++ {
		tree.Insert(i)
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated with sorted insert")
	}
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated with sorted insert")
	}
	maxExpected := int(1.44*math.Log2(17) + 1)
	if tree.Height() > maxExpected {
		t.Fatalf("height %d too large for 15 elements (max %d)", tree.Height(), maxExpected)
	}
}

func TestCloneCreatesIndependentCopy(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	clone := tree.Clone()
	if clone.Size() != 3 {
		t.Fatalf("expected clone size 3, got %d", clone.Size())
	}
	if !clone.Contains(50) || !clone.Contains(30) || !clone.Contains(70) {
		t.Fatal("clone should contain all values")
	}
	if clone.Height() != tree.Height() {
		t.Fatal("clone height should match original")
	}
}

func TestInsertToOriginalDoesntAffectClone(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	clone := tree.Clone()
	tree.Insert(30)
	if tree.Size() != 2 {
		t.Fatal("expected original size 2")
	}
	if clone.Size() != 1 {
		t.Fatal("expected clone size 1")
	}
	if clone.Contains(30) {
		t.Fatal("clone should not contain 30")
	}
}

func TestRemoveFromOriginalDoesntAffectClone(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	clone := tree.Clone()
	tree.Remove(30)
	if tree.Contains(30) {
		t.Fatal("original should not contain 30")
	}
	if !clone.Contains(30) {
		t.Fatal("clone should contain 30")
	}
}

func TestSingleElementTree(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(42)
	v, _ := tree.Min()
	if v != 42 {
		t.Fatal("expected min 42")
	}
	v, _ = tree.Max()
	if v != 42 {
		t.Fatal("expected max 42")
	}
	if !tree.Contains(42) {
		t.Fatal("expected to contain 42")
	}
	tree.Remove(42)
	if !tree.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestReverseSortedInsertProducesBalancedTree(t *testing.T) {
	tree := avl.New[int]()
	for i := 15; i >= 1; i-- {
		tree.Insert(i)
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated with reverse sorted insert")
	}
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated with reverse sorted insert")
	}
	maxExpected := int(1.44*math.Log2(17) + 1)
	if tree.Height() > maxExpected {
		t.Fatalf("height %d too large for 15 elements (max %d)", tree.Height(), maxExpected)
	}
}

func TestLargeNumberOfElements(t *testing.T) {
	tree := avl.New[int]()
	for i := 1; i <= 1000; i++ {
		tree.Insert(i)
	}
	if tree.Size() != 1000 {
		t.Fatalf("expected size 1000, got %d", tree.Size())
	}
	v, _ := tree.Min()
	if v != 1 {
		t.Fatal("expected min 1")
	}
	v, _ = tree.Max()
	if v != 1000 {
		t.Fatal("expected max 1000")
	}
	maxExpected := int(1.44*math.Log2(1002) + 1)
	if tree.Height() > maxExpected {
		t.Fatalf("height %d exceeds expected max %d for 1000 elements", tree.Height(), maxExpected)
	}
}

func TestNegativeNumbers(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(-10)
	tree.Insert(0)
	tree.Insert(10)
	tree.Insert(-20)
	v, _ := tree.Min()
	if v != -20 {
		t.Fatal("expected min -20")
	}
	v, _ = tree.Max()
	if v != 10 {
		t.Fatal("expected max 10")
	}
	if !tree.Contains(-10) {
		t.Fatal("expected to contain -10")
	}
	if !verifyBSTProperty(tree) {
		t.Fatal("BST property violated")
	}
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated")
	}
}

func TestRemoveAllElementsOneByOne(t *testing.T) {
	tree := avl.New[int]()
	tree.Insert(50)
	tree.Insert(30)
	tree.Insert(70)
	tree.Insert(20)
	tree.Insert(40)
	tree.Remove(20)
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated after removing 20")
	}
	tree.Remove(40)
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated after removing 40")
	}
	tree.Remove(30)
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated after removing 30")
	}
	tree.Remove(70)
	if !verifyAVLBalance(tree) {
		t.Fatal("AVL balance violated after removing 70")
	}
	tree.Remove(50)
	if !tree.IsEmpty() {
		t.Fatal("expected empty")
	}
}

func TestAlternatingInsertRemoveMaintainsBalance(t *testing.T) {
	tree := avl.New[int]()
	for i := 1; i <= 50; i++ {
		tree.Insert(i)
		if i%5 == 0 {
			tree.Remove(i - 2)
		}
		if !verifyAVLBalance(tree) {
			t.Fatalf("AVL balance violated at iteration %d", i)
		}
	}
}

func TestWorksWithStrings(t *testing.T) {
	tree := avl.New[string]()
	tree.Insert("banana")
	tree.Insert("apple")
	tree.Insert("cherry")
	v, _ := tree.Min()
	if v != "apple" {
		t.Fatalf("expected min apple, got %s", v)
	}
	v, _ = tree.Max()
	if v != "cherry" {
		t.Fatalf("expected max cherry, got %s", v)
	}
	result := tree.InOrder()
	expected := []string{"apple", "banana", "cherry"}
	for i := range expected {
		if result[i] != expected[i] {
			t.Fatalf("index %d: expected %s, got %s", i, expected[i], result[i])
		}
	}
}

func TestWorksWithFloats(t *testing.T) {
	tree := avl.New[float64]()
	tree.Insert(3.14)
	tree.Insert(1.41)
	tree.Insert(2.71)
	v, _ := tree.Min()
	if v != 1.41 {
		t.Fatalf("expected min 1.41, got %f", v)
	}
	v, _ = tree.Max()
	if v != 3.14 {
		t.Fatalf("expected max 3.14, got %f", v)
	}
}

func TestInOrderIsSorted(t *testing.T) {
	tree := avl.New[int]()
	values := []int{50, 25, 75, 10, 30, 60, 90, 5, 15}
	for _, v := range values {
		tree.Insert(v)
	}
	result := tree.InOrder()
	sorted := make([]int, len(values))
	copy(sorted, values)
	sort.Ints(sorted)
	for i := range sorted {
		if result[i] != sorted[i] {
			t.Fatalf("index %d: expected %d, got %d", i, sorted[i], result[i])
		}
	}
}
