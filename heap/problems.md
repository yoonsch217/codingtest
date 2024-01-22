
### 378. Kth Smallest Element in a Sorted Matrix

https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix

문제: 이차원 매트릭스의 각 행과 열이 정렬이 되어 있다. 전체 숫자 중 작은 것부터 순서대로 셌을 때 k 번째인 값을 구하라.

**minheap**   
external sort처럼 생각을 해보면, 각 행이 정렬되어 있기 때문에 각 행의 맨 앞끼리만 비교하면서 k번째인 값을 구하면 된다.   
리스트가 두 개라면 포인터를 하나씩 두면서 할 수 있지만 N개의 포인터를 두는 건 쉽지 않다.   
이럴 때는 힙을 사용하면 쉽게 해결할 수 있다.    
힙에 각 `(row의 head 값, row, col)` 를 넣고 k 번 iterate하면 된다.   

Time: O(k * logN + N), Space: O(N)


<details>

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


**Maxheap approach**   

size k max heap 을 만들어서 전체 element를 다 넣어도 된다. 그러면 k smallest가 남게 된다. 그 중 max가 답이므로 heappop 한 번 하면 된다.   
Time: O(M * N * logk), Space: O(k)


**Binary Search approach**   

- matrix의 최솟값과 최댓값 사이를 binary search 한다.
- 어떤 값 x에 대해서 matrix에 존재하는 x보다 작거나 같은 값의 수를 구하는 함수를 count_less_or_equal() 라고 하자.
- count_less_or_equal(x) = k 를 만족하는 최소의 x를 구해야한다. 
  - 최소의 x를 구해야하는 이유는, count_less_or_equal(x) != count_less_or_equal(x-1) 라는 거고, 이는 즉 x가 matrix에 실제 존재하는 값이라는 뜻이다.
  - 각 column과 row는 sorted 상태이다. 따라서 count_less_or_equal를 구할 때 이를 활용한다. 
  - row_pointer는 0부터 증가하고 col_pointer는 n-1로 세팅한다. 각 row에서 맨 뒤부터 보면서 x보다 작거나 같은 수가 나올 때 그 row에 대한 count를 알 수 있다. 다음 row에서는 해당 col부터 시작하면 된다.
  - https://leetcode.com/problems/search-a-2d-matrix-ii/ 참고



<details>

    
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

        return ans
```

</details>


Complexity
- Time: O((M+N) * logD) for D difference between max and min. 
   - logD는 min에서 max 사이를 binary search하는 시간이다. count_less_or_equal은 O(M+N)이다. top-right corner부터 시작해서 가로로 최대 N번, 세로로 최대 M번 이동할 때까지 찾기 때문이다.
- Space: O(1) 



