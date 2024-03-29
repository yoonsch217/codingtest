### 378. Kth Smallest Element in a Sorted Matrix

https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix

문제: 이차원 매트릭스의 각 행과 열이 정렬이 되어 있다. 전체 숫자 중 작은 것부터 순서대로 셌을 때 k 번째인 값을 구하라.


<details><summary>Approach 1</summary>

**minheap**   

external sort처럼 생각을 해보면, 각 행이 정렬되어 있기 때문에 각 행의 맨 앞끼리만 비교하면서 k번째인 값을 구하면 된다.   
리스트가 두 개라면 포인터를 하나씩 두면서 할 수 있지만 N개의 포인터를 두는 건 쉽지 않다.   
이럴 때는 힙을 사용하면 쉽게 해결할 수 있다.    
힙에 각 `(row의 head 값, row, col)` 를 넣고 k 번 iterate하면 된다.   

- Time: O(k * logN + N)
- Space: O(N)

```python
def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
    heap = []
    n = len(matrix)
    for i in range(n):
        heapq.heappush(heap, (matrix[i][0], i, 0))
    
    for _ in range(k):
        ans, _row, _col = heapq.heappop(heap)
        _next_col = _col + 1
        if _next_col >= n:
            continue
        heapq.heappush(heap, (matrix[_row][_next_col], _row, _next_col))
    return ans
```

</details>


<details><summary>Approach 2</summary>


**Maxheap approach**   

size k max heap 을 만들어서 전체 element를 다 넣어도 된다. 그러면 k smallest가 남게 된다. 그 중 max가 답이므로 heappop 한 번 하면 된다.   

- Time: O(M * N * logk)
- Space: O(k)

minheap에 비해 시간은 더 걸리고 메모리는 덜 쓴다.


</details>

<details><summary>Approach 3</summary>

**Binary Search approach**   

- matrix의 최솟값과 최댓값 사이를 binary search 한다.
- 어떤 값 x에 대해서 matrix에 존재하는 x보다 작거나 같은 값의 수를 구하는 함수를 count_less_or_equal() 라고 하자.
- count_less_or_equal(x) = k 를 만족하는 최소의 x를 구해야한다. 
  - 최소의 x를 구해야하는 이유는, count_less_or_equal(x) != count_less_or_equal(x-1) 라는 거고, 이는 즉 x가 matrix에 실제 존재하는 값이라는 뜻이다.
  - 각 column과 row는 sorted 상태이다. 따라서 count_less_or_equal를 구할 때 이를 활용한다. 
  - row_pointer는 0부터 증가하고 col_pointer는 n-1로 세팅한다. 각 row에서 맨 뒤부터 보면서 x보다 작거나 같은 수가 나올 때 그 row에 대한 count를 알 수 있다. 다음 row에서는 해당 col부터 시작하면 된다.
  - https://leetcode.com/problems/search-a-2d-matrix-ii/ 참고


```
binary search 조건: mid 보다 작거나 같은 값이 k개 이상일 경우
x x x x o o o
        ^
처음으로 이 값보다 작거나 같은 값이 k개이다. 즉, 이 값이 matrix에 존재한다. 그리고 본인 포함해서 작은 k개가 있다.
left를 반환하면 된다.
```

```py
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])  # For general, a matrix need not to be a square

        def count_less_or_equal(x):
            cnt = 0
            col_idx = n - 1  # start with the rightmost column of the first row
            for row_idx in range(m):
                while col_idx >= 0 and matrix[row_idx][col_idx] > x: 
                    col_idx -= 1  # decrease column until matrix[r][c] <= x
                cnt += (col_idx + 1)
            return cnt

        left, right = matrix[0][0], matrix[-1][-1]
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if count_less_or_equal(mid) >= k:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans  # or 그냥 ans 사용하지 않고 left를 반환해도 된다.
```

Complexity
- Time: O((M+N) * logD) for D difference between max and min. 
   - logD는 min에서 max 사이를 binary search하는 시간이다. count_less_or_equal은 O(M+N)이다. top-right corner부터 시작해서 가로로 최대 N번, 세로로 최대 M번 이동할 때까지 찾기 때문이다.
- Space: O(1) 

</details>














### 1834. Single-Threaded CPU

https://leetcode.com/problems/single-threaded-cpu/description/

문제: You are given n​​​​​​ tasks labeled from 0 to n - 1 represented by a 2D integer array tasks, where tasks[i] = [enqueueTimei, processingTimei] means that the i​​​​​​th​​​​ task will be available to process at enqueueTimei and will take processingTimei to finish processing.    
You have a single-threaded CPU that can process at most one task at a time and will act in the following way:    
- If the CPU is idle and there are no available tasks to process, the CPU remains idle.
- If the CPU is idle and there are available tasks, the CPU will choose the one with the shortest processing time. If multiple tasks have the same shortest processing time, it will choose the task with the smallest index.
- Once a task is started, the CPU will process the entire task without stopping.
- The CPU can finish a task then start a new one instantly.    
Return the order in which the CPU will process the tasks.

<details><summary>Approach 1</summary>

beat 99.29%!!

- 모든 task에 대해서 enqueue 시간 순서대로 먼저 정렬을 한다.
- current time을 초기화한다. 0으로 해도 되고 가장 빠른 task enqueue 시간으로 해도 된다. 방어 로직이 있다.
- sorted tasks에 대해 index를 두고 `while i < n` 조건으로 iterate한다.
- sorted tasks에서 current time 이하의 task를 heap에 넣는다. 넣을 땐 (process time, task id) 순서로 넣는다.
- 만약 current time보다 작은 task enqueue가 없다면(heap이 비어있다면) index를 증가시키고 다시 iterate한다.
- heap에 데이터가 있다면 heappop을 한다. 가장 process time이 작은 task가 나올 것이다.
- current time을 업데이트하고 다음 iterate를 진행한다.
- task index가 n이 됐다면(모든 task가 heap에 들어갔다면) heap 순서대로 뽑으면 된다.



```py
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        heap, res = [], []
        n = len(tasks)

        for i in range(n):
            tasks[i].append(i)  # add task id
        tasks.sort()

        i = 0
        cur_time = tasks[0][0]
        while i < n:
            while i < n and tasks[i][0] <= cur_time:
                heapq.heappush(heap, (tasks[i][1], tasks[i][2], tasks[i][0]))
                i += 1
            if not heap:
                """
                이게 있어야 empty heap에 대해 대응할 수 있다. 
                처음에는 두 while 사이에 넣었는데 그러면 안 된다. 
                proc_time으로 시간이 잘 흘렀는데 heap에 넣기 전에 다시 과거로 돌아갈 수도 있기 때문이다. 
                여기에 넣어야 heap에 넣으려고 시도해봤는데도 empty heap이 되는 경우만 다음 task로 건너뛰어준다.
                """
                cur_time = tasks[i][0]  
                continue
            proc_time, task_id, _ = heapq.heappop(heap)
            res.append(task_id)
            cur_time += proc_time

        while heap:
            res.append(heapq.heappop(heap)[1])
        
        return res

```

</details>
