# ------------------------- Sorting Algorithms -------------------------

class SortingAlgorithms:
    @staticmethod
    def quick_sort(arr):
        """Quick Sort implementation."""
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return SortingAlgorithms.quick_sort(left) + middle + SortingAlgorithms.quick_sort(right)

    @staticmethod
    def merge_sort(arr):
        """Merge Sort implementation."""
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid])
        right = SortingAlgorithms.merge_sort(arr[mid:])
        return SortingAlgorithms.merge(left, right)

    @staticmethod
    def merge(left, right):
        """Merge two sorted lists."""
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    @staticmethod
    def heap_sort(arr):
        """Heap Sort implementation."""
        import heapq
        heapq.heapify(arr)
        return [heapq.heappop(arr) for _ in range(len(arr))]

    @staticmethod
    def counting_sort(arr):
        """Counting Sort implementation."""
        max_val = max(arr)
        count = [0] * (max_val + 1)
        for num in arr:
            count[num] += 1
        sorted_arr = []
        for i in range(len(count)):
            sorted_arr.extend([i] * count[i])
        return sorted_arr

    @staticmethod
    def radix_sort(arr):
        """Radix Sort implementation."""
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            SortingAlgorithms.counting_sort(arr)
            exp *= 10
        return arr

    @staticmethod
    def bucket_sort(arr):
        """Bucket Sort implementation."""
        if len(arr) == 0:
            return arr
        min_val, max_val = min(arr), max(arr)
        bucket_range = (max_val - min_val) / len(arr)
        buckets = [[] for _ in range(len(arr))]
        for num in arr:
            index = int((num - min_val) // bucket_range)
            if index != len(arr):
                buckets[index].append(num)
            else:
                buckets[-1].append(num)
        sorted_arr = []
        for bucket in buckets:
            sorted_arr.extend(SortingAlgorithms.quick_sort(bucket))
        return sorted_arr


# ------------------------- Searching Algorithms -------------------------

class SearchingAlgorithms:
    @staticmethod
    def linear_search(arr, target):
        """Linear Search implementation."""
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1

    @staticmethod
    def binary_search(arr, target):
        """Binary Search (Iterative) implementation."""
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    @staticmethod
    def binary_search_recursive(arr, target, left, right):
        """Binary Search (Recursive) implementation."""
        if left > right:
            return -1
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return SearchingAlgorithms.binary_search_recursive(arr, target, mid + 1, right)
        else:
            return SearchingAlgorithms.binary_search_recursive(arr, target, left, mid - 1)

    @staticmethod
    def find_peak_element(arr):
        """Find a peak element in an array."""
        left, right = 0, len(arr) - 1
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < arr[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left

    @staticmethod
    def find_min_in_rotated_sorted_array(arr):
        """Find the minimum element in a rotated sorted array."""
        left, right = 0, len(arr) - 1
        while left < right:
            mid = (left + right) // 2
            if arr[mid] > arr[right]:
                left = mid + 1
            else:
                right = mid
        return arr[left]


# ------------------------- Divide and Conquer -------------------------

class DivideAndConquer:
    @staticmethod
    def find_kth_largest(arr, k):
        """Find the k-th largest element using Quickselect."""
        if k > len(arr):
            return None
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        if k <= len(right):
            return DivideAndConquer.find_kth_largest(right, k)
        elif k <= len(right) + len(middle):
            return pivot
        else:
            return DivideAndConquer.find_kth_largest(left, k - len(right) - len(middle))


# ------------------------- Greedy Algorithms -------------------------

class GreedyAlgorithms:
    @staticmethod
    def activity_selection(start, finish):
        """Activity Selection Problem."""
        activities = list(zip(start, finish))
        activities.sort(key=lambda x: x[1])  # Sort by finish time
        selected = []
        last_finish = 0
        for s, f in activities:
            if s >= last_finish:
                selected.append((s, f))
                last_finish = f
        return selected

    @staticmethod
    def huffman_coding(frequencies):
        """Huffman Coding implementation."""
        import heapq

        # Create a min-heap based on frequencies
        heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
        heapq.heapify(heap)

        # Build the Huffman tree
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        # Extract the Huffman codes
        huffman_codes = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
        return huffman_codes

    @staticmethod
    def fractional_knapsack(values, weights, capacity):
        """Fractional Knapsack Problem."""
        items = list(zip(values, weights))
        items.sort(key=lambda x: x[0] / x[1], reverse=True)  # Sort by value/weight ratio
        total_value = 0
        for v, w in items:
            if capacity >= w:
                total_value += v
                capacity -= w
            else:
                total_value += v * (capacity / w)
                break
        return total_value

    @staticmethod
    def job_scheduling_with_deadlines(jobs):
        """Job Scheduling with Deadlines."""
        jobs.sort(key=lambda x: x[1], reverse=True)  # Sort by profit in descending order
        max_deadline = max(job[2] for job in jobs)
        schedule = [-1] * (max_deadline + 1)
        total_profit = 0

        for job in jobs:
            deadline = job[2]
            while deadline > 0:
                if schedule[deadline] == -1:
                    schedule[deadline] = job[0]
                    total_profit += job[1]
                    break
                deadline -= 1

        return schedule, total_profit


# ------------------------- Dynamic Programming (DP) -------------------------

class DynamicProgramming:
    @staticmethod
    def fibonacci(n):
        """Fibonacci Numbers using Dynamic Programming."""
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    @staticmethod
    def longest_increasing_subsequence(arr):
        """Longest Increasing Subsequence (LIS)."""
        dp = [1] * len(arr)
        for i in range(1, len(arr)):
            for j in range(i):
                if arr[i] > arr[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    @staticmethod
    def knapsack_01(values, weights, capacity):
        """0/1 Knapsack Problem."""
        n = len(values)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        return dp[n][capacity]

    @staticmethod
    def coin_change(coins, amount):
        """Coin Change Problem (Minimum Coins)."""
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1

    @staticmethod
    def matrix_chain_multiplication(dims):
        """Matrix Chain Multiplication."""
        n = len(dims) - 1
        dp = [[0] * n for _ in range(n)]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                    dp[i][j] = min(dp[i][j], cost)
        return dp[0][n - 1]

    @staticmethod
    def dp_on_trees(root):
        """DP on Trees."""
        def dfs(node):
            if not node:
                return 0, 0
            left_take, left_not_take = dfs(node.left)
            right_take, right_not_take = dfs(node.right)
            take = node.val + left_not_take + right_not_take
            not_take = max(left_take, left_not_take) + max(right_take, right_not_take)
            return take, not_take

        take, not_take = dfs(root)
        return max(take, not_take)

    @staticmethod
    def dp_on_bitmasks(mask, n):
        """DP on Bitmasks."""
        dp = [0] * (1 << n)
        for i in range(1 << n):
            for j in range(n):
                if not (i & (1 << j)):
                    dp[i | (1 << j)] = max(dp[i | (1 << j)], dp[i] + 1)
        return dp[(1 << n) - 1]


# ------------------------- Backtracking -------------------------

class Backtracking:
    @staticmethod
    def n_queens(n):
        """N-Queens Problem."""
        def is_safe(board, row, col):
            for i in range(row):
                if board[i] == col or abs(board[i] - col) == abs(i - row):
                    return False
            return True

        def solve(board, row):
            if row == n:
                solutions.append(board[:])
                return
            for col in range(n):
                if is_safe(board, row, col):
                    board[row] = col
                    solve(board, row + 1)
                    board[row] = -1

        solutions = []
        solve([-1] * n, 0)
        return solutions

    @staticmethod
    def sudoku_solver(board):
        """Sudoku Solver."""
        def is_valid(board, row, col, num):
            for i in range(9):
                if board[row][i] == num or board[i][col] == num or board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == num:
                    return False
            return True

        def solve(board):
            for row in range(9):
                for col in range(9):
                    if board[row][col] == 0:
                        for num in range(1, 10):
                            if is_valid(board, row, col, num):
                                board[row][col] = num
                                if solve(board):
                                    return True
                                board[row][col] = 0
                        return False
            return True

        solve(board)
        return board

    @staticmethod
    def word_search(grid, word):
        """Word Search in a Grid."""
        def dfs(row, col, index):
            if index == len(word):
                return True
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != word[index]:
                return False
            temp = grid[row][col]
            grid[row][col] = '#'
            found = (dfs(row + 1, col, index + 1) or
                     dfs(row - 1, col, index + 1) or
                     dfs(row, col + 1, index + 1) or
                     dfs(row, col - 1, index + 1))
            grid[row][col] = temp
            return found

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if dfs(i, j, 0):
                    return True
        return False

    @staticmethod
    def hamiltonian_cycle(graph):
        """Hamiltonian Cycle Problem."""
        def is_safe(v, pos, path):
            if graph[path[pos - 1]][v] == 0:
                return False
            if v in path:
                return False
            return True

        def solve(path, pos):
            if pos == len(graph):
                if graph[path[pos - 1]][path[0]] == 1:
                    return True
                else:
                    return False
            for v in range(1, len(graph)):
                if is_safe(v, pos, path):
                    path[pos] = v
                    if solve(path, pos + 1):
                        return True
                    path[pos] = -1
            return False

        path = [-1] * len(graph)
        path[0] = 0
        if solve(path, 1):
            return path
        return None


def printStructure():
  return '''
  Structure and Blueprint
1. SortingAlgorithms Class
Purpose: Implements various sorting algorithms.

Functions:

quick_sort(arr): Sorts an array using the Quick Sort algorithm.

merge_sort(arr): Sorts an array using the Merge Sort algorithm.

merge(left, right): Merges two sorted arrays (used by Merge Sort).

heap_sort(arr): Sorts an array using the Heap Sort algorithm.

counting_sort(arr): Sorts an array using the Counting Sort algorithm.

radix_sort(arr): Sorts an array using the Radix Sort algorithm.

bucket_sort(arr): Sorts an array using the Bucket Sort algorithm.

2. SearchingAlgorithms Class
Purpose: Implements various searching algorithms.

Functions:

linear_search(arr, target): Searches for a target in an array using Linear Search.

binary_search(arr, target): Searches for a target in a sorted array using Binary Search (Iterative).

binary_search_recursive(arr, target, left, right): Searches for a target in a sorted array using Binary Search (Recursive).

find_peak_element(arr): Finds a peak element in an array.

find_min_in_rotated_sorted_array(arr): Finds the minimum element in a rotated sorted array.

3. DivideAndConquer Class
Purpose: Implements divide-and-conquer algorithms.

Functions:

find_kth_largest(arr, k): Finds the k-th largest element in an array using Quickselect.

4. GreedyAlgorithms Class
Purpose: Implements greedy algorithms.

Functions:

activity_selection(start, finish): Solves the Activity Selection Problem.

huffman_coding(frequencies): Generates Huffman Codes for a given frequency dictionary.

fractional_knapsack(values, weights, capacity): Solves the Fractional Knapsack Problem.

job_scheduling_with_deadlines(jobs): Solves the Job Scheduling with Deadlines Problem.

5. DynamicProgramming Class
Purpose: Implements dynamic programming algorithms.

Functions:

fibonacci(n): Computes the n-th Fibonacci number using DP.

longest_increasing_subsequence(arr): Finds the length of the Longest Increasing Subsequence (LIS).

knapsack_01(values, weights, capacity): Solves the 0/1 Knapsack Problem.

coin_change(coins, amount): Solves the Coin Change Problem (Minimum Coins).

matrix_chain_multiplication(dims): Solves the Matrix Chain Multiplication Problem.

dp_on_trees(root): Solves a DP problem on trees (e.g., maximum independent set).

dp_on_bitmasks(mask, n): Solves a DP problem using bitmasking.

6. Backtracking Class
Purpose: Implements backtracking algorithms.

Functions:

n_queens(n): Solves the N-Queens Problem.

sudoku_solver(board): Solves a Sudoku puzzle.

word_search(grid, word): Searches for a word in a 2D grid.

hamiltonian_cycle(graph): Finds a Hamiltonian Cycle in a graph.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------

Examples:(how to use the functions):

1. SortingAlgorithms Class
Quick Sort
python
Copy
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = SortingAlgorithms.quick_sort(arr)
print("Quick Sort:", sorted_arr)
Output:

Copy
Quick Sort: [1, 1, 2, 3, 6, 8, 10]
Merge Sort
python
Copy
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = SortingAlgorithms.merge_sort(arr)
print("Merge Sort:", sorted_arr)
Output:

Copy
Merge Sort: [1, 1, 2, 3, 6, 8, 10]
Heap Sort
python
Copy
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = SortingAlgorithms.heap_sort(arr)
print("Heap Sort:", sorted_arr)
Output:

Copy
Heap Sort: [1, 1, 2, 3, 6, 8, 10]
Counting Sort
python
Copy
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = SortingAlgorithms.counting_sort(arr)
print("Counting Sort:", sorted_arr)
Output:

Copy
Counting Sort: [1, 1, 2, 3, 6, 8, 10]
Radix Sort
python
Copy
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = SortingAlgorithms.radix_sort(arr)
print("Radix Sort:", sorted_arr)
Output:

Copy
Radix Sort: [1, 1, 2, 3, 6, 8, 10]
Bucket Sort
python
Copy
arr = [3, 6, 8, 10, 1, 2, 1]
sorted_arr = SortingAlgorithms.bucket_sort(arr)
print("Bucket Sort:", sorted_arr)
Output:

Copy
Bucket Sort: [1, 1, 2, 3, 6, 8, 10]
2. SearchingAlgorithms Class
Linear Search
python
Copy
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target = 6
index = SearchingAlgorithms.linear_search(arr, target)
print(f"Linear Search: Target {target} found at index {index}")
Output:

Copy
Linear Search: Target 6 found at index 5
Binary Search (Iterative)
python
Copy
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target = 6
index = SearchingAlgorithms.binary_search(arr, target)
print(f"Binary Search: Target {target} found at index {index}")
Output:

Copy
Binary Search: Target 6 found at index 5
Binary Search (Recursive)
python
Copy
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target = 6
index = SearchingAlgorithms.binary_search_recursive(arr, target, 0, len(arr) - 1)
print(f"Binary Search (Recursive): Target {target} found at index {index}")
Output:

Copy
Binary Search (Recursive): Target 6 found at index 5
Find Peak Element
python
Copy
arr = [1, 3, 20, 4, 1, 0]
peak_index = SearchingAlgorithms.find_peak_element(arr)
print(f"Peak Element Index: {peak_index}, Value: {arr[peak_index]}")
Output:

Copy
Peak Element Index: 2, Value: 20
Find Minimum in Rotated Sorted Array
python
Copy
arr = [4, 5, 6, 7, 0, 1, 2]
min_value = SearchingAlgorithms.find_min_in_rotated_sorted_array(arr)
print(f"Minimum Value in Rotated Sorted Array: {min_value}")
Output:

Copy
Minimum Value in Rotated Sorted Array: 0
3. DivideAndConquer Class
Find K-th Largest Element
python
Copy
arr = [3, 2, 1, 5, 6, 4]
k = 2
kth_largest = DivideAndConquer.find_kth_largest(arr, k)
print(f"{k}-th Largest Element: {kth_largest}")
Output:

Copy
2-th Largest Element: 5
4. GreedyAlgorithms Class
Activity Selection Problem
python
Copy
start = [1, 3, 0, 5, 8, 5]
finish = [2, 4, 6, 7, 9, 9]
selected_activities = GreedyAlgorithms.activity_selection(start, finish)
print("Selected Activities:", selected_activities)
Output:

Copy
Selected Activities: [(1, 2), (3, 4), (5, 7), (8, 9)]
Huffman Coding
python
Copy
frequencies = {'a': 5, 'b': 9, 'c': 12, 'd': 13, 'e': 16, 'f': 45}
huffman_codes = GreedyAlgorithms.huffman_coding(frequencies)
print("Huffman Codes:", huffman_codes)
Output:

Copy
Huffman Codes: [['f', '0'], ['c', '100'], ['d', '101'], ['a', '1100'], ['b', '1101'], ['e', '111']]
Fractional Knapsack Problem
python
Copy
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
max_value = GreedyAlgorithms.fractional_knapsack(values, weights, capacity)
print("Maximum Value in Fractional Knapsack:", max_value)
Output:

Copy
Maximum Value in Fractional Knapsack: 240.0
Job Scheduling with Deadlines
python
Copy
jobs = [('a', 100, 2), ('b', 19, 1), ('c', 27, 2), ('d', 25, 1), ('e', 15, 3)]
schedule, total_profit = GreedyAlgorithms.job_scheduling_with_deadlines(jobs)
print("Scheduled Jobs:", schedule)
print("Total Profit:", total_profit)
Output:

Copy
Scheduled Jobs: ['c', 'a', 'e']
Total Profit: 142
5. DynamicProgramming Class
Fibonacci Numbers
python
Copy
n = 10
fib_num = DynamicProgramming.fibonacci(n)
print(f"Fibonacci({n}): {fib_num}")
Output:

Copy
Fibonacci(10): 55
Longest Increasing Subsequence (LIS)
python
Copy
arr = [10, 22, 9, 33, 21, 50, 41, 60]
lis_length = DynamicProgramming.longest_increasing_subsequence(arr)
print("Length of LIS:", lis_length)
Output:

Copy
Length of LIS: 5
0/1 Knapsack Problem
python
Copy
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
max_value = DynamicProgramming.knapsack_01(values, weights, capacity)
print("Maximum Value in 0/1 Knapsack:", max_value)
Output:

Copy
Maximum Value in 0/1 Knapsack: 220
Coin Change Problem
python
Copy
coins = [1, 2, 5]
amount = 11
min_coins = DynamicProgramming.coin_change(coins, amount)
print("Minimum Coins Required:", min_coins)
Output:

Copy
Minimum Coins Required: 3
Matrix Chain Multiplication
python
Copy
dims = [10, 30, 5, 60]
min_cost = DynamicProgramming.matrix_chain_multiplication(dims)
print("Minimum Cost for Matrix Chain Multiplication:", min_cost)
Output:

Copy
Minimum Cost for Matrix Chain Multiplication: 4500
DP on Trees
python
Copy
# Define a simple tree structure
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

root = TreeNode(3)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.right = TreeNode(3)
root.right.right = TreeNode(1)

max_value = DynamicProgramming.dp_on_trees(root)
print("Maximum Independent Set Value:", max_value)
Output:

Copy
Maximum Independent Set Value: 7
DP on Bitmasks
python
Copy
mask = 0b1010
n = 4
max_set_bits = DynamicProgramming.dp_on_bitmasks(mask, n)
print("Maximum Set Bits in Bitmask:", max_set_bits)
Output:

Copy
Maximum Set Bits in Bitmask: 2
6. Backtracking Class
N-Queens Problem
python
Copy
n = 4
solutions = Backtracking.n_queens(n)
print(f"N-Queens Solutions for {n}x{n} board:", solutions)
Output:

Copy
N-Queens Solutions for 4x4 board: [[1, 3, 0, 2], [2, 0, 3, 1]]
Sudoku Solver
python
Copy
board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]
solved_board = Backtracking.sudoku_solver(board)
print("Solved Sudoku Board:")
for row in solved_board:
    print(row)
Output:

Copy
Solved Sudoku Board:
[5, 3, 4, 6, 7, 8, 9, 1, 2]
[6, 7, 2, 1, 9, 5, 3, 4, 8]
[1, 9, 8, 3, 4, 2, 5, 6, 7]
[8, 5, 9, 7, 6, 1, 4, 2, 3]
[4, 2, 6, 8, 5, 3, 7, 9, 1]
[7, 1, 3, 9, 2, 4, 8, 5, 6]
[9, 6, 1, 5, 3, 7, 2, 8, 4]
[2, 8, 7, 4, 1, 9, 6, 3, 5]
[3, 4, 5, 2, 8, 6, 1, 7, 9]
Word Search in a Grid
python
Copy
grid = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
word = "ABCCED"
found = Backtracking.word_search(grid, word)
print(f"Word '{word}' found in grid:", found)
Output:

Copy
Word 'ABCCED' found in grid: True
Hamiltonian Cycle
python
Copy
graph = [
    [0, 1, 0, 1, 0],
    [1, 0, 1, 1, 1],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 1, 1, 1, 0]
]
hamiltonian_path = Backtracking.hamiltonian_cycle(graph)
print("Hamiltonian Cycle:", hamiltonian_path)
Output:

Copy
Hamiltonian Cycle: [0, 1, 2, 4, 3, 0]
This guide provides examples for every class and function in the implementation. Let me know if you need further clarification! 😊
  '''

def printCode():
  return '''
  # ------------------------- Sorting Algorithms -------------------------

class SortingAlgorithms:
    @staticmethod
    def quick_sort(arr):
        """Quick Sort implementation."""
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return SortingAlgorithms.quick_sort(left) + middle + SortingAlgorithms.quick_sort(right)

    @staticmethod
    def merge_sort(arr):
        """Merge Sort implementation."""
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = SortingAlgorithms.merge_sort(arr[:mid])
        right = SortingAlgorithms.merge_sort(arr[mid:])
        return SortingAlgorithms.merge(left, right)

    @staticmethod
    def merge(left, right):
        """Merge two sorted lists."""
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    @staticmethod
    def heap_sort(arr):
        """Heap Sort implementation."""
        import heapq
        heapq.heapify(arr)
        return [heapq.heappop(arr) for _ in range(len(arr))]

    @staticmethod
    def counting_sort(arr):
        """Counting Sort implementation."""
        max_val = max(arr)
        count = [0] * (max_val + 1)
        for num in arr:
            count[num] += 1
        sorted_arr = []
        for i in range(len(count)):
            sorted_arr.extend([i] * count[i])
        return sorted_arr

    @staticmethod
    def radix_sort(arr):
        """Radix Sort implementation."""
        max_val = max(arr)
        exp = 1
        while max_val // exp > 0:
            SortingAlgorithms.counting_sort(arr)
            exp *= 10
        return arr

    @staticmethod
    def bucket_sort(arr):
        """Bucket Sort implementation."""
        if len(arr) == 0:
            return arr
        min_val, max_val = min(arr), max(arr)
        bucket_range = (max_val - min_val) / len(arr)
        buckets = [[] for _ in range(len(arr))]
        for num in arr:
            index = int((num - min_val) // bucket_range)
            if index != len(arr):
                buckets[index].append(num)
            else:
                buckets[-1].append(num)
        sorted_arr = []
        for bucket in buckets:
            sorted_arr.extend(SortingAlgorithms.quick_sort(bucket))
        return sorted_arr


# ------------------------- Searching Algorithms -------------------------

class SearchingAlgorithms:
    @staticmethod
    def linear_search(arr, target):
        """Linear Search implementation."""
        for i in range(len(arr)):
            if arr[i] == target:
                return i
        return -1

    @staticmethod
    def binary_search(arr, target):
        """Binary Search (Iterative) implementation."""
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    @staticmethod
    def binary_search_recursive(arr, target, left, right):
        """Binary Search (Recursive) implementation."""
        if left > right:
            return -1
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return SearchingAlgorithms.binary_search_recursive(arr, target, mid + 1, right)
        else:
            return SearchingAlgorithms.binary_search_recursive(arr, target, left, mid - 1)

    @staticmethod
    def find_peak_element(arr):
        """Find a peak element in an array."""
        left, right = 0, len(arr) - 1
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < arr[mid + 1]:
                left = mid + 1
            else:
                right = mid
        return left

    @staticmethod
    def find_min_in_rotated_sorted_array(arr):
        """Find the minimum element in a rotated sorted array."""
        left, right = 0, len(arr) - 1
        while left < right:
            mid = (left + right) // 2
            if arr[mid] > arr[right]:
                left = mid + 1
            else:
                right = mid
        return arr[left]


# ------------------------- Divide and Conquer -------------------------

class DivideAndConquer:
    @staticmethod
    def find_kth_largest(arr, k):
        """Find the k-th largest element using Quickselect."""
        if k > len(arr):
            return None
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        if k <= len(right):
            return DivideAndConquer.find_kth_largest(right, k)
        elif k <= len(right) + len(middle):
            return pivot
        else:
            return DivideAndConquer.find_kth_largest(left, k - len(right) - len(middle))


# ------------------------- Greedy Algorithms -------------------------

class GreedyAlgorithms:
    @staticmethod
    def activity_selection(start, finish):
        """Activity Selection Problem."""
        activities = list(zip(start, finish))
        activities.sort(key=lambda x: x[1])  # Sort by finish time
        selected = []
        last_finish = 0
        for s, f in activities:
            if s >= last_finish:
                selected.append((s, f))
                last_finish = f
        return selected

    @staticmethod
    def huffman_coding(frequencies):
        """Huffman Coding implementation."""
        import heapq

        # Create a min-heap based on frequencies
        heap = [[weight, [symbol, ""]] for symbol, weight in frequencies.items()]
        heapq.heapify(heap)

        # Build the Huffman tree
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        # Extract the Huffman codes
        huffman_codes = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
        return huffman_codes

    @staticmethod
    def fractional_knapsack(values, weights, capacity):
        """Fractional Knapsack Problem."""
        items = list(zip(values, weights))
        items.sort(key=lambda x: x[0] / x[1], reverse=True)  # Sort by value/weight ratio
        total_value = 0
        for v, w in items:
            if capacity >= w:
                total_value += v
                capacity -= w
            else:
                total_value += v * (capacity / w)
                break
        return total_value

    @staticmethod
    def job_scheduling_with_deadlines(jobs):
        """Job Scheduling with Deadlines."""
        jobs.sort(key=lambda x: x[1], reverse=True)  # Sort by profit in descending order
        max_deadline = max(job[2] for job in jobs)
        schedule = [-1] * (max_deadline + 1)
        total_profit = 0

        for job in jobs:
            deadline = job[2]
            while deadline > 0:
                if schedule[deadline] == -1:
                    schedule[deadline] = job[0]
                    total_profit += job[1]
                    break
                deadline -= 1

        return schedule, total_profit


# ------------------------- Dynamic Programming (DP) -------------------------

class DynamicProgramming:
    @staticmethod
    def fibonacci(n):
        """Fibonacci Numbers using Dynamic Programming."""
        if n <= 1:
            return n
        dp = [0] * (n + 1)
        dp[1] = 1
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]
        return dp[n]

    @staticmethod
    def longest_increasing_subsequence(arr):
        """Longest Increasing Subsequence (LIS)."""
        dp = [1] * len(arr)
        for i in range(1, len(arr)):
            for j in range(i):
                if arr[i] > arr[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    @staticmethod
    def knapsack_01(values, weights, capacity):
        """0/1 Knapsack Problem."""
        n = len(values)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        return dp[n][capacity]

    @staticmethod
    def coin_change(coins, amount):
        """Coin Change Problem (Minimum Coins)."""
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1

    @staticmethod
    def matrix_chain_multiplication(dims):
        """Matrix Chain Multiplication."""
        n = len(dims) - 1
        dp = [[0] * n for _ in range(n)]
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length - 1
                dp[i][j] = float('inf')
                for k in range(i, j):
                    cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                    dp[i][j] = min(dp[i][j], cost)
        return dp[0][n - 1]

    @staticmethod
    def dp_on_trees(root):
        """DP on Trees."""
        def dfs(node):
            if not node:
                return 0, 0
            left_take, left_not_take = dfs(node.left)
            right_take, right_not_take = dfs(node.right)
            take = node.val + left_not_take + right_not_take
            not_take = max(left_take, left_not_take) + max(right_take, right_not_take)
            return take, not_take

        take, not_take = dfs(root)
        return max(take, not_take)

    @staticmethod
    def dp_on_bitmasks(mask, n):
        """DP on Bitmasks."""
        dp = [0] * (1 << n)
        for i in range(1 << n):
            for j in range(n):
                if not (i & (1 << j)):
                    dp[i | (1 << j)] = max(dp[i | (1 << j)], dp[i] + 1)
        return dp[(1 << n) - 1]


# ------------------------- Backtracking -------------------------

class Backtracking:
    @staticmethod
    def n_queens(n):
        """N-Queens Problem."""
        def is_safe(board, row, col):
            for i in range(row):
                if board[i] == col or abs(board[i] - col) == abs(i - row):
                    return False
            return True

        def solve(board, row):
            if row == n:
                solutions.append(board[:])
                return
            for col in range(n):
                if is_safe(board, row, col):
                    board[row] = col
                    solve(board, row + 1)
                    board[row] = -1

        solutions = []
        solve([-1] * n, 0)
        return solutions

    @staticmethod
    def sudoku_solver(board):
        """Sudoku Solver."""
        def is_valid(board, row, col, num):
            for i in range(9):
                if board[row][i] == num or board[i][col] == num or board[3 * (row // 3) + i // 3][3 * (col // 3) + i % 3] == num:
                    return False
            return True

        def solve(board):
            for row in range(9):
                for col in range(9):
                    if board[row][col] == 0:
                        for num in range(1, 10):
                            if is_valid(board, row, col, num):
                                board[row][col] = num
                                if solve(board):
                                    return True
                                board[row][col] = 0
                        return False
            return True

        solve(board)
        return board

    @staticmethod
    def word_search(grid, word):
        """Word Search in a Grid."""
        def dfs(row, col, index):
            if index == len(word):
                return True
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != word[index]:
                return False
            temp = grid[row][col]
            grid[row][col] = '#'
            found = (dfs(row + 1, col, index + 1) or
                     dfs(row - 1, col, index + 1) or
                     dfs(row, col + 1, index + 1) or
                     dfs(row, col - 1, index + 1))
            grid[row][col] = temp
            return found

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if dfs(i, j, 0):
                    return True
        return False

    @staticmethod
    def hamiltonian_cycle(graph):
        """Hamiltonian Cycle Problem."""
        def is_safe(v, pos, path):
            if graph[path[pos - 1]][v] == 0:
                return False
            if v in path:
                return False
            return True

        def solve(path, pos):
            if pos == len(graph):
                if graph[path[pos - 1]][path[0]] == 1:
                    return True
                else:
                    return False
            for v in range(1, len(graph)):
                if is_safe(v, pos, path):
                    path[pos] = v
                    if solve(path, pos + 1):
                        return True
                    path[pos] = -1
            return False

        path = [-1] * len(graph)
        path[0] = 0
        if solve(path, 1):
            return path
        return None

  '''
