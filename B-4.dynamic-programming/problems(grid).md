

### 63. Unique Paths II

https://leetcode.com/problems/unique-paths-ii/

문제: robot이 m x n matrix의 제일 왼쪽 위에 놓여져있고 오른쪽이나 아래로만 움직일 수 있다. matrix[i][j]의 값이 1이라면 그 곳은 로봇이 움직일 수 없다. grid의 제일 오른쪽 아래에 갈 수 있는 경로의 수를 구하라.



<details><summary>Approach 1</summary>

```
dp(i, j): Number of the unique paths from (0, 0) to (i, j)
dp(i, j) is 
- 0 if (i, j) is out of the grid
- 0 if (i, j) is an obstacle
- 1 if (i, j) is the top-left corner
- dp(i-1, j) + dp(i, j-1) otherwise
```


top down

```python
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if  obstacleGrid[0][0] == 1:  # 이 부분을 빠뜨리지 말자. 답도 틀리고 수행 시간도 길어진다.
            return 0

        @lru_cache(maxsize=None)
        def getPathHelper(i, j):
            if i == 0 and j == 0:
                return 1
            if not (0 <= i < m and 0 <= j < n):
                return 0
            if obstacleGrid[i][j] == 1:
                return 0
            return getPathHelper(i-1, j) + getPathHelper(i, j-1)
        return getPathHelper(m-1, n-1)
```

bottom up

```python
class Solution:
    def uniquePathsWithObstacles(self, grid: List[List[int]]) -> int:
        num_rows = len(grid)
        num_cols = len(grid[0])

        dp = [[0 for _ in range(num_cols)] for _ in range(num_rows)]  # num_rows 가 뒤에 loop 에 있어야 한다.

        if grid[0][0] == 1:
            return 0
        dp[0][0] = 1

        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] == 1:
                    continue
                if r != 0:
                    dp[r][c] += dp[r-1][c]
                if c != 0:
                    dp[r][c] += dp[r][c-1]
        
        return dp[num_rows-1][num_cols-1]



```

```python
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if  obstacleGrid[0][0] == 1:  # 이 부분을 빠뜨리지 말자. 답도 틀리고 수행 시간도 길어진다.
            return 0

        prev_row = [0] * n
        prev_row[0] = 1

        for i in range(m):
            cur_row = [0] * n
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    cur_row[j] = 0
                    continue
                cur_row[j] = cur_row[j-1] + prev_row[j]
            prev_row = cur_row
        
        return prev_row[n-1]
```

</details>









### 64. Minimum Path Sum

https://leetcode.com/problems/minimum-path-sum/description/

문제: m x n grid에서 non-negative 숫자로 채워져있다. top left에서 right bottom으로 가야하는데 오른쪽 혹은 아래로만 움직일 수 있다. 가는 길에 있는 숫자의 합이 최소가 되도록 가라.



<details><summary>Approach 1</summary>

```
dp(i, j): The minimum sum of costs to reach (i, j) from the top left corner
dp(i, j) is 
 - grid[i][j] if i == 0 and j == 0
 - inf if (i, j) is out of the grid
 - min(dp(i-1, j), dp(i, j-1)) + grid[i][j] otherwise
```

- top down: recursion 방식. O(mn) / O(mn)    
- bottom up: iterative하게 하려면 하나의 row를 저장하면서 하는 방법이 있다. O(mn) / O(n)    
- optimization on space: bottom up을 하면서 original matrix에 업데이트하는 방법도 있다. O(mn) / O(1)


top down

```py
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        @lru_cache(maxsize=None)
        def getPathHelper(i, j):
            if i == 0 and j == 0:
                return grid[i][j]
            if not (0 <= i < m and 0 <= j < n):
                return math.inf
            return min(getPathHelper(i-1, j), getPathHelper(i, j-1)) + grid[i][j]
        
        return getPathHelper(m-1, n-1)
```

bottom up with updating the original grid

```py
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] = grid[i][j-1] + grid[i][j]
                elif j == 0:
                    grid[i][j] = grid[i-1][j] + grid[i][j]
                else:
                    grid[i][j] = min(grid[i-1][j], grid[i][j-1]) + grid[i][j]
        return grid[m-1][n-1]
```



</details>










### 221. Maximal Square

https://leetcode.com/problems/maximal-square/

문제: mxn binary matrix가 0 혹은 1로 채워져있다. 1로만 이루어진 가장 큰 정사각형의 넓이를 반환하라.


<details><summary>Approach 1</summary>

- 어떤 꼭지점 (i,j) 를 기준으로 왼쪽, 위, 왼쪽위 점들이 둘러싸는 점들이다.
- 왼쪽 점 (i, j-1), 위쪽 점 (i-1, j) 이 겹치는 부분은 현재 점을 기준으로도 연장될 수가 있다.
- 만약, 4, 4 라면 현재 점 기준으로 왼쪽 4개, 위쪽 4개를 더 포함할 수 있다는 건데, 제일 왼쪽 위 꼭지점은 아직 알 수 없다.
- (i-1, j-1) 도 만약 4라면 제일 왼쪽 위 꼭지점도 포함한다는 뜻이다. 왜나하면 바로 왼쪽 점인 (i, j-1) 과 동일하게 왼쪽으로 뻗어나가는데 한 칸 위까지 뻗어나가기 때문이다.
  - dp(i-1, j-1) 은 dp(i-1, j) 에서 가로축 시작이 동일하고 세로축이 하나 올라간 것이고, dp(i, j-1)과 세로축 시작이 동일하고 가로축이 하나 옮겨진 것이다. 따라서 딱 모서리가 cover된다.

```
dp(i, j): matrix[i][j] 위치를 오른쪽 아래 꼭지점으로 두어서 왼쪽 위로 만들 수 있는 최대의 정사각형의 한 변 길이    
dp(i, j) = min(dp(i-1,j), dp(i,j-1), dp(i-1,j-1)) + 1 
```

이렇게 하면 dp로 풀이는 가능하고, 공간 최적화를 하려면 직전 row의 정보만 보관하면 된다.



```py
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        n_row = len(matrix)
        n_col = len(matrix[0])
        
        prev_row = [0] * n_col
        cur_row = [0] * n_col

        max_side = 0

        for i in range(n_row):
            for j in range(n_col):
                if matrix[i][j] != '1':
                    continue
                if j == 0:
                    cur_row[j] = 1  # 여기서 max_side 업데이트 하지 않고 continue로 넘어가버려서 틀렸었다. 
                else:
                    cur_row[j] = min(cur_row[j-1], prev_row[j], prev_row[j-1]) + 1
                max_side = max(max_side, cur_row[j])
            prev_row = cur_row  # 밑에서 cur_row가 바라보는 객체를 다시 만들어주니까 deepcopy 없이 그냥 prev_row가 바라보는 객체만 바꿔주면 된다.
            cur_row = [0] * n_col
        
        return max_side * max_side
```

prev_row, cur_row 두 개를 쓰는 게 아니라 prev_row, left_value 이렇게 두 개를 쓰려고 해봤다.    
그런데 row를 오른쪽으로 이동하면서 prev_row의 자기 위치를 업데이트해야하는데 그렇게 하면 (i-1, j-1) 위치를 구하기가 어렵다.     
왜냐하면 prev_row[j-1]은 left_value와 동일하기 때문이다.    
그냥 row 두 개를 쓰자.   

아니면 아래처럼 row 하나만 쓰기도 했다.

```python
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        n_rows, n_cols = len(matrix), len(matrix[0])
        prev_rows = [0] * n_cols
        largest = 0

        for i in range(n_rows):
            for j in range(n_cols):
                if matrix[i][j] == '0':
                    prev_rows[j] = 0
                    continue

                if i == 0 or j == 0 or prev_rows[j-1] == 0 or prev_rows[j] == 0:
                    prev_rows[j] = 1
                elif prev_rows[j-1] == prev_rows[j]:
                    if matrix[i-prev_rows[j-1]][j-prev_rows[j-1]] == '1':
                        prev_rows[j] = prev_rows[j-1] + 1
                    else:
                        prev_rows[j] = prev_rows[j-1]
                else:
                    prev_rows[j] = 1 + min(prev_rows[j-1], prev_rows[j])
                largest = max(largest, prev_rows[j])
            
        return largest * largest
                    
```


</details>









### 931. Minimum Falling Path Sum


https://leetcode.com/problems/minimum-falling-path-sum/description/

문제: n x n matrix가 있을 때 falling path 중 minimum sum을 구하라. falling path란 제일 윗 row에서 제일 밑 row 까지 내려오는데 내려올 때 바로 아래나 대각선 아래로만 내려오는 path를 의미한다.

<details><summary>Approach 1</summary>

```
dp(i, j): minimum falliing path sum to get to matrix[i][j]
dp(i, j) = matrix[i][j] + min(dp(i-1, j-1), dp(i-1, j), dp(i-1, j+1))
```

grid 문제를 연속으로 푸니까 기본 문제는 똑같은 틀에서 벗어나질 않네.


이것도 마찬가지로 bottom up으로 할 수 있는데 O(N) space로 할 수 있다. 그냥 밑에 row부터 차례대로 올라오는 것이다. 결국 row 0 의 결괏값만 알면 되는 건데 이는 row 1의 결괏값만 필요하다.


```py
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)

        @lru_cache(maxsize=None)
        def helper(i, j):
            if not (0 <= i < n and 0 <= j < n):
                return math.inf
            if i == 0:
                return matrix[i][j]
            return matrix[i][j] + min(helper(i-1, j-1), min(helper(i-1, j), helper(i-1, j+1)))
        
        res = math.inf
        for j in range(n):
            res = min(res, helper(n-1, j))
        return res
```

</details>


