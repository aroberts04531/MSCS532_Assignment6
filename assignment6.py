"""
Assignment 6 - Medians and Order Statistics and Elementary Data Structures
Author: Alexandria Roberts
Course: MSCS 532

This file implements:
- Deterministic selection (Median of Medians)
- Randomized selection (Quickselect)
- Simple timing tests for both
- Basic array/matrix operations
- Stack, queue (array-based)
- Singly linked list
"""

import random 
import time
from typing import List, Any, Optional

# ============================
# Part 1: Selection Algorithms
# ============================

def deterministic_select(arr: List[int], k: int) -> int:
    """Median-of-medians select. Returns k -th smallest (1-based)."""

    if not 1 <= k <= len(arr):
        raise ValueError("k must be between 1 and len(arr)")

    def select(a: List[int], index: int) -> int:
         if len(a) <= 5:
            a_sorted = sorted(a)
            return a_sorted[index]
    
         groups = [a[i:i + 5] for i in range (0, len(a), 5)]
         medians = [sorted(g)[len(g) // 2] for g in groups]

         pivot = select(medians, len(medians) // 2)

         lows = [x for x in a if x < pivot]
         equals = [x for x in a if x == pivot]
         highs = [x for x in a if x > pivot]

         if index < len(lows):
             return select(lows, index)
         elif index < len(lows) + len(equals):
              return pivot
         else:
              return select(highs, index - len(lows) - len(equals))
    
    return select(list(arr), k -1)

def randomized_select(arr: List[int], k: int) -> int:
    """Randomized Quickselect. Returns k -th smallest (1-based)."""

    if not 1 <= k <= len(arr):
        raise ValueError("k must be between 1 and len(arr)")
   
    a = list(arr)
    target = k - 1

    def partition(left: int, right: int, pivot_index: int) -> int:
        pivot = a[pivot_index]
        a[pivot_index], a[right] = a[right], a[pivot_index]
        store = left 
        for i in range(left, right):
            if a[i] < pivot:
                a[store], a[i] = a[i], a[store]
                store += 1
        a[store], a[right] = a[right], a[store]
        return store
    
    def quickselect(left: int, right: int, index: int) -> int:
        if left == right:
            return a[left]
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)

        if index == pivot_index:
            return a[pivot_index]
        elif index < pivot_index:
            return quickselect(left, pivot_index - 1, index)
        else:
            return quickselect(pivot_index + 1, right, index)
    return quickselect(0, len(a) -1, target)

def generate_array(n: int, distribution: str) -> List[int]:
    if distribution == "random":
        return [random.randint(0, n) for _ in range(n)]
    elif distribution == "sorted":
        return list(range(n))
    elif distribution == "reverse":
        return list(range(n, 0, -1))
    else: 
        raise ValueError("Unknown distribution")

def time_selection(func, arr: List[int], k: int, repeats: int = 3) -> float:
    total = 0.0
    for _ in range(repeats):
        data = list(arr)
        start = time.perf_counter()
        _ = func(data, k)
        end = time.perf_counter()
        total += (end - start)
    return total / repeats

def run_selection_experiments():
    size = [1000, 5000]
    dists = ["random", "sorted", "reverse"]
    algs = [
        ("Deterministic", deterministic_select),
        ("randomized", randomized_select),
    ]

    print("=== Selection Algorithm Experiments ===")
    for n in size:
        for dist in dists:
            arr = generate_array(n, dist)
            k = n // 2
            print(f"\nSize={n}, dist={dist}, k={k}")
            for name, func in algs:
                avg = time_selection(func, arr, k)
                print(f" {name}: {avg:.6f} seconds")


# ========================================
# Part 2: Elementary Data Structures (min)
# ========================================

# ---- Arrays and matrices using lists ----

def array_insert(a: List[Any], index: int, value: Any) -> None:
    """Inserts values at index in array a (list). O(n -index)."""
    a.insert(index, value)

def array_delete(a: List[Any], index: int) -> Any:
    """Delete and returns element at index in array a (list). O(n - index)."""
    return a.pop(index)

def array_access(a: List[Any], index: int) -> Any:
    """Return element at index. O(1)"""
    return a[index]

def matrix_get(m: List[List[Any]], row: int, col: int) -> Any:
    """Get element at (row, col). O(1)."""
    return m[row][col]

def matrix_set(m: List[List[Any]], row: int, col: int, value: Any) -> None:
    """Set element at (row, col) to value. O(1)."""
    m[row][col] = value 

# -----------Stack (array-based)---------------

class Stack:
    """Stack using Python list as array."""

    def __init__(self):
        self.data: List[Any] = []

    def push(self, item: Any) -> None:
        self.data.append(item) 

    def pop(self) -> Any:
        if not self.data:
            raise IndexError("pop from empty stack")
        return self.data.pop()
    
    def peek(self) -> Any:
        if not self.data:
            raise IndexError("peek from empty stack")
        return self.data[-1]
    def is_empty(self) -> bool:
        return len(self.data) == 0
    
# ------------- Queue (array-based) ----------------

class Queue:
    """
    Simple queue using list.
    Note: dequeue using pop(0) is O(n), which you can mention in analysis.
    """
    
    def __init__(self):
        self.data: List[Any] = []

    def enqueue(self, item: Any) -> None:
        self.data.append(item)

    def dequeue(self) -> Any:
        if not self.data:
            raise IndexError("dequeue from empty queue")
        return self.data.pop(0)
    
    def peek(self) -> Any:
        if not self.data:
            raise IndexError("peek from empty queue")
        return self.data[0]
    
    def is_empty(self) -> bool:
        return len(self.data) == 0
    
# -------Singly Linked List -----------------

class ListNode:
    def __init__(self, value: Any, next: Optional["ListNode"] = None):
        self.value = value
        self.next = next

class SinglyLinkedList:
    def __init__(self):
        self.head: Optional[ListNode] = None
    
    def insert_at_head(self, value: Any) -> None:
        self.head = ListNode(value, self.head)
    
    def insert_at_tail(self, value: Any) -> None:
        new_node = ListNode(value)
        if self.head is None:
            self.head = new_node
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node

    def delete_value(self, value: Any) -> bool:
        curr = self.head
        prev = None
        while curr:
            if curr.value == value:
                if prev is None:
                    self.head = curr.next
                else:
                    prev.next = curr.next
                return True
            prev = curr
            curr = curr.next
        return False

    def traverse(self) -> List[Any]:
        result = []
        curr = self.head
        while curr:
            result.append(curr.value)
            curr = curr.next
        return result

# ------------ Quick sanity tests ---------------

if __name__ == "__main__":
    nums = [2, 4, 6, 8, 9]
    print("Array:", nums)
    print("3rd smallest (deterministic):", deterministic_select(nums, 3))
    print("3rd smallest (randomized):", randomized_select(nums, 3))

    run_selection_experiments()

    a = [1, 2, 3]
    array_insert(a, 1, 99)
    print("\nArray after ops:", a, "access index 1:", array_access(a, 1))

    m = [[0, 0], [0, 0]]
    matrix_set(m, 0, 1, 5)
    print("Matrix:", m, "get(0,1):", matrix_get(m, 0, 1))

    s = Stack()
    s.push(1); s.push(2); s.push(3)
    print("Stack pop:", s.pop())

    q = Queue()
    q.enqueue("a"); q.enqueue("b"); q.enqueue("c")
    print("Queue dequeue:", q.dequeue())

    ll = SinglyLinkedList()
    ll.insert_at_head(1)
    ll.insert_at_tail(2)
    ll.insert_at_tail(3)
    ll.delete_value(2)
    print("Linked list traverse:", ll.traverse())

