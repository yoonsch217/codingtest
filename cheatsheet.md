# 코딩 인터뷰 Cheat Sheet

## 📋 기본 자료구조 및 알고리즘

### Array & String
- **Sliding Window**: O(N) 시간에 부분 배열/문자열 문제 해결
  - Two pointers로 window 크기 조절
  - 구현 원리: left, right 포인터로 window 범위 조절하며 조건 만족하는 최적의 window 찾기
  - 사용 시점: 연속된 부분 배열/문자열의 최대/최소, 개수 세기 문제
  - 예: Longest Substring Without Repeating, Fruit Into Baskets
- **Two Pointers**: 정렬된 배열에서 효율적인 탐색
  - 구현 원리: 정렬된 양 끝에서 시작하여 합/차이에 따라 포인터 이동
  - 사용 시점: 정렬된 배열에서 합/차이/거리 관련 문제
  - 예: 3Sum, Container With Most Water
- **Prefix Sum**: 구간 합 빠르게 계산 O(1)
  - 구현 원리: 누적 합 배열 미리 계산, 구간 합 = prefix[j] - prefix[i-1]
  - 사용 시점: 반복적인 구간 합 계산이 필요한 문제
- **Dutch National Flag**: 세 가지 값으로 분류 (Sort Colors)
  - 구현 원리: low, mid, high 세 포인터로 0,1,2 구역 분할하며 정렬
  - 사용 시점: 세 가지 값으로 분류/정렬해야 하는 O(N) 문제

### Stack & Queue
- **Monotonic Stack**: 감소/증가 순서 유지
  - 구현 원리: 스택에 증가하는 순서 유지, 현재 원소보다 큰 원소들 pop하며 처리
  - 사용 시점: 다음 큰/작은 원소 찾기, 범위 계산 문제
  - Daily Temperatures: 뒤에서부터 탐색 O(N)
  - Trapping Rain Water: 웅덩이 층별 계산
- **Monotonic Queue**: Sliding Window Maximum
  - 구현 원리: deque 사용, window 내 최대값 유지하며 오래된/작은 원소 제거
  - 사용 시점: sliding window 내 최대/최소값 빠르게 찾기
- **Min Stack**: 추가 스택으로 최솟값 관리 O(1)
  - 구현 원리: 보조 스택에 현재까지의 최솟값들 저장, pop 시 동기화
  - 사용 시점: 스택 연산 중 최솟값을 O(1)에 조회해야 할 때

### Hash Table
- **Anagram 그룹화**: 문자열 정렬 또는 문자 빈도수
  - 구현 원리: 정렬된 문자열이나 문자 빈도수를 key로 사용하여 같은 그룹끼리 묶기
  - 사용 시점: 문자열 그룹화, 빠른 검색, 중복 제거 필요 시

### Binary Search
- **Pattern**: `oooxxx` 형태에서 경계 찾기
  - 구현 원리: 조건 만족/불만족 구간에서 중간값으로 경계 좁혀나가기
  - 사용 시점: 정렬된 데이터에서 특정 조건의 경계값 찾기
  - left: 조건 만족하지 않는 최소 index
  - right: 조건 만족하는 최대 index
- **응용**: K closest elements, rotated array 최솟값
- find_first 문제
  - 구현 원리: 조건을 만족하는 경우 res에 임시 저장하고 다시 right = mid - 1 로 옮겨서 진행하기
  - 사용 시점: 조건을 만족하는 첫 번째/가장 왼쪽 원소 찾기

### Graph
- **BFS**: 최단 경로, 레벨 탐색 O(V+E)
  - 구현 원리: queue 사용, 레벨별 탐색하며 최단 경로 보장
  - 사용 시점: 최단 경로, 레벨 탐색, 연결성 확인
- **DFS**: 사이클 감지, 위상 정렬
  - 구현 원리: stack/recursion 사용, 깊이 우선 탐색하며 경로 추적
  - 사용 시점: 사이클 감지, 경로 찾기, 위상 정렬
- **Dijkstra**: 최단 경로 (양수 가중치) O(E log V)
  - 구현 원리: priority queue 사용, 최단 거리 확정된 노드부터 처리
  - 사용 시점: 양수 가중치 그래프에서 최단 경로
- **Bellman-Ford**: 음수 가중치, K stops 제한 O(VE)
  - 구현 원리: 모든 간선 V-1번 relaxation, 음수 사이클 감지 가능
  - 사용 시점: 음수 가중치, K stops 제한, 음수 사이클 감지
- **Topological Sort**: Kahn's Algorithm (in-degree), DFS (white/gray/black)
  - 구현 원리: in-degree 0인 노드부터 순서대로 제거하거나 DFS 후위 순회
  - 사용 시점: 작업 순서, 의존성 관계, 사이클 없는 방향 그래프
- **MST**: Kruskal (Union-Find), Prim
  - 구현 원리: Kruskal은 가중치 순 정렬 후 Union-Find로 사이클 방지, Prim은 임의 노드부터 최소 간선 선택
  - 사용 시점: 최소 비용으로 모든 노드 연결, 네트워크 설계

### Tree
- **BST**: 삽입/삭제 O(log n)
  - 구현 원리: 왼쪽 < 현재 < 오른쪽 속성 유지하며 재귀적으로 탐색
  - 사용 시점: 동적 집합, 빠른 검색/삽입/삭제 필요 시
- **Morris Traversal**: O(N)/O(1) 공간 순회
  - 구현 원리: 중위 후계자 포인터 조작으로 스택 없이 트리 순회, 현재 노드의 오른쪽 자식이 null이면 오른쪽으로 이동, 아니면 중위 후계자 찾아서 오른쪽 포인터를 현재 노드로 연결
  - 사용 시점: 공간 제한이 심할 때 트리 순회
- **Trie**: 문자열 검색, 접두사
  - 구현 원리: 각 노드가 문자를 저장, 공통 접두사 공유하여 검색 효율화
  - 사용 시점: 문자열 검색, 자동완성, 접두사 관련 문제

### Heap
- **우선순위 큐**: 최대/최솟값 빠른 접근
  - 구현 원리: 완전 이진 트리, 부모-자식 관계로 우선순위 유지
  - 사용 시점: 동적으로 최대/최소값을 자주 조회/추가할 때
- **Top K Frequent using heap**: O(N log K)
  - 구현 원리: 빈도수 계산 후 max-heap에 (빈도수, 원소) 저장, K번 pop
  - 사용 시점: 빈도수 기반 상위 K개 원소 찾기
- **Top K Frequent using bucket sort**: O(N)
  - 구현 원리: 빈도수를 인덱스로 하는 bucket 배열, 높은 빈도부터 K개 선택. bucket 배열의 크기는 N을 넘지 않는 것을 활용한 방법
  - 사용 시점: 빈도수 기반 상위 K개 원소 찾기

### Dynamic Programming
- **State Reduction**: 최근 n개만 필요한지 확인
  - 구현 원리: DP 테이블 전체 대신 필요한 상태만 저장하여 공간 최적화
  - 사용 시점: DP 테이블이 너무 클 때 공간 최적화 필요
- **Kadane's Algorithm**: 최대 부분합 O(N)
  - 구현 원리: 현재까지의 최대 부분합과 현재 원소만으로 시작하는 부분합 비교 `cur_sum = max(num, cur_sum + num)`
  - 사용 시점: 최대 부분합, 최대 부분곱 등 연속된 구간 최적화
- **Knapsack**: 조합 문제
  - 구현 원리: 물건별로 포함/미포함 경우 고려하여 최적값 갱신
  - 사용 시점: 무게/용량 제한 하의 최대 가치/조합 문제
  - **0/1 Knapsack**: 각 물건은 한 번만 사용 가능, 역순 탐색
  - **Unbounded Knapsack**: 동전 등 중복 사용 가능, 정순 탐색
- **State Machine**: 여러 상태 간 전환 (Stock problems)
  - 구현 원리: 각 시점별 상태(보유/미보유)를 DP로 관리하며 최적화
  - 사용 시점: 여러 상태 간 전환, 제약 조건이 있는 최적화

### Greedy
- **매 선택이 최적해로 이어짐**
  - 구현 원리: 현재 최적 선택이 전체 최적해를 보장하는 증명 기반 접근
  - 사용 시점: 현재 선택이 미래에 영향을 주지 않는 최적화 문제
- **Activity Selection, Fractional Knapsack**
  - 구현 원리: 종료시간/단위무게 가치 기준 정렬 후 greedy 선택
  - 사용 시점: 활동 선택, 분할 가능한 배낭 문제

### Bit Manipulation
- **기본 연산**: `&`, `|`, `^`, `<<`, `>>`, `~`
  - 구현 원리: 비트 단위 연산으로 플래그, 마스크, 상태 표현
  - 사용 시점: 플래그 관리, 마스킹, 상태 표현
- **Bit Mask**: 정수로 상태 표현
  - 구현 원리: 각 비트를 특정 상태나 플래그로 사용하여 메모리 효율화
  - 사용 시점: 여러 상태를 하나의 정수로 표현, 부분 집합 열거
- **n & (n-1)**: 가장 낮은 1 비트 제거
  - 구현 원리: n의 2진표현에서 가장 오른쪽 1을 0으로 만드는 효과
  - 사용 시점: 1의 개수 세기, 2의 거듭제곤 확인

## 🎯 핵심 패턴 & 문제 유형

### 1. Subarray Problems
**문제**: 배열에서 연속된 subarray의 합이 가장 큰 값을 구하라 (Maximum Subarray)
```python
# Maximum Subarray (Kadane)
max_sum = cur_sum = nums[0]
for num in nums[1:]:
    cur_sum = max(num, cur_sum + num)
    max_sum = max(max_sum, cur_sum)
```
    
# Sliding Window Fixed Size
**문제**: 크기가 k인 sliding window를 움직이며 각 window의 최대값들을 배열로 구하라
**시간복잡도**: O(N*k) - 각 window마다 max() 함수 호출
```python
from collections import deque
window = deque()
result = []
for i in range(len(nums)):
    window.append(nums[i])
    if i >= k-1:
        result.append(max(window))  # O(k) for each window
        window.popleft()
return result

# 더 효율적인 방법: Monotonic Queue 사용하면 O(N) 가능
```

### 2. Two Pointers

**문제**: 정렬된 배열에서 합이 target이 되는 두 원소 찾기 (Two Sum)
```python
# Sorted Array Two Sum
left, right = 0, len(nums)-1
while left < right:
    s = nums[left] + nums[right]
    if s == target: return [left, right]
    elif s < target: left += 1
    else: right -= 1
```

**문제**: 배열에서 조건을 만족하는 가장 긴 subarray의 길이 찾기 (Variable Sliding Window)
```python
# Sliding Window Variable Size
left = 0
max_len = 0
for right in range(len(nums)):
    # expand window
    while condition_violated:
        # shrink window
        left += 1
    max_len = max(max_len, right - left + 1)
return max_len
```

### 3. Monotonic Stack
**문제**: 각 원소에 대해 다음으로 큰 원소 찾기 (Next Greater Element)
```python
stack = []  # stores indices
result = [-1] * len(nums)  # default if no greater element
for i, val in enumerate(nums):
    while stack and nums[stack[-1]] < val:
        prev = stack.pop()
        result[prev] = val  # current val is next greater for prev
    stack.append(i)
return result
```

### 4. Monotonic Queue
**문제**: 크기가 k인 sliding window를 움직이며 각 window의 최대값들을 O(N)에 구하라 (Sliding Window Maximum)
```python
from collections import deque
window = deque()
result = []
for i, val in enumerate(nums):
    # Remove elements smaller than current
    while window and window[-1][0] <= val:
        window.pop()
    window.append((val, i))
    
    # Remove elements outside window
    while window and window[0][1] <= i - k:
        window.popleft()
    
    if i >= k - 1:
        result.append(window[0][0])  # largest element is located at head
return result
```

### 5. BFS Shortest Path

**문제**: 그래프에서 시작점에서 도착점까지의 최단 경로 길이 찾기
```python
from collections import deque
queue = deque([(start, 0)])
visited = {start}
while queue:
    node, dist = queue.popleft()
    for neighbor in graph[node]:
        if neighbor == target: return dist + 1
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append((neighbor, dist + 1))
return -1  # no path found
```

### 6. DP Patterns

**문제**: 주식을 여러 번 거래할 수 있지만, 매도 후 1일 쿨다운 필요 (Best Time to Buy and Sell Stock with Cooldown)
**점화식**: 
- dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i]) (i일에 주식 미보유)
- dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i]) (i일에 주식 보유, 쿨다운 고려)
```python
# State Machine DP - Stock with Cooldown
n = len(prices)
if n <= 1: return 0
dp = [[0, 0] for _ in range(n)]
dp[0][1] = -prices[0]  # buy on day 0

for i in range(1, n):
    dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])  # sell or stay
    if i >= 2:
        dp[i][1] = max(dp[i-1][1], dp[i-2][0] - prices[i])  # buy with cooldown
    else:
        dp[i][1] = max(dp[i-1][1], -prices[i])  # buy on day 1

return dp[-1][0]  # max profit without holding stock
```

**문제**: 2D 격자에서 (0,0)에서 (n-1,m-1)까지의 최대 합 경로값 찾기
**점화식**: dp[i][j] = grid[i][j] + max(dp[i-1][j], dp[i][j-1]) (최대 경로 합)
```python
# 2D DP with Space Optimization
prev = [0] * m
for i in range(n):
    cur = [0] * m
    for j in range(m):
        if i == 0 and j == 0: cur[j] = grid[i][j]
        elif i == 0: cur[j] = cur[j-1] + grid[i][j]
        elif j == 0: cur[j] = prev[j] + grid[i][j]
        else: cur[j] = max(prev[j], cur[j-1]) + grid[i][j]
    prev = cur
return prev[-1]
```

**문제**: 동전들을 사용하여 target 금액을 만들 수 있는 조합의 수 구하기, 동전은 중복 사용 가능 (Unbounded Knapsack)
**점화식**: dp[amount] = dp[amount] + dp[amount - coin] (조합의 수)
- dp[amount] 값을 업데이트하기 위해 내부 loop에서 amount를 돌 때, 정방향으로 돌아야한다. 
- 그래야 dp[amount-coin] 을 참조할 때 coin 이 사용된 dp 값을 바라보기 때문에 중복 사용 가능의 조건을 만족한다. 
```python
# Knapsack (Combination)
dp = [0] * (target + 1)
dp[0] = 1
for coin in coins:
    for amount in range(coin, target + 1):
        dp[amount] += dp[amount - coin]
return dp[target]
```

**문제**: 동전들을 사용하여 target 금액을 만들 수 있는 조합의 수 구하기, 동전은 중복 사용 불가능 (0/1 Knapsack)
**점화식**: dp[amount] = dp[amount] + dp[amount - coin] (조합의 수)
- dp[amount] 값을 업데이트하기 위해 내부 loop에서 amount를 돌 때, 역방향으로 돌아야한다. 
- 그래야 dp[amount-coin] 을 참조할 때 coin 이 사용되지 않은 dp 값을 바라보기 때문에 중복이 없다.
- 
```python
# 0/1 Knapsack (조합의 수, 중복 불가)
dp = [0] * (target + 1)
dp[0] = 1 # 아무것도 고르지 않는 경우 1가지

for coin in coins:
    # 0/1 Knapsack: 역방향 루프
    for amount in range(target, coin - 1, -1):
        dp[amount] += dp[amount - coin]

return dp[target]
```


**문제**: 각 물건에는 weight와 value가 있다. 각 물건은 한 번만 사용 가능하여 capacity 내에서 최대 가치를 주는 조합 구하기 (0/1 Knapsack)
**점화식**: dp[w] = max(dp[w], dp[w-weight] + value) (최대 가치)
```python
# 0/1 Knapsack (No repetition)
dp = [0] * (capacity + 1)
for weight, value in range(len(items)):
    # 중복이 안 되니까 역방향 루프 사용 
    for w in range(capacity, weight - 1, -1):
        dp[w] = max(dp[w], dp[w - weight] + value)
return dp[capacity]
```

**문제**: 동전들을 사용하여 target 금액을 만들 수 있는 순열의 수 구하기, 동전은 중복 사용 가능 (Unbounded Permutation)
**점화식**: dp[amount] = Sum of dp[amount - coin] for coin in coins
- coin iteration을 안쪽 루프에서 해야하고, amount iteration 을 바깥 쪽 루프에서 한다.
```python
# Unbounded Knapsack (Permutation)
dp = [0] * (target + 1)
dp[0] = 1
# 순열을 위해 금액(amount)이 외부 루프로 나옴
for amount in range(1, target + 1):
    for coin in coins:
        if amount >= coin:
            # 현재 금액을 만들기 위해 '마지막에 사용한 동전'이 무엇인지에 따라 모든 경우의 수를 누적함
            dp[amount] += dp[amount - coin]
return dp[target]
```

**문제**: 동전들을 사용하여 target 금액을 만들 수 있는 순열의 수 구하기, 동전은 중복 사용 불가능 (Permutation, Traveling Salesman Problem)
**점화식**: dp[state][amount]: 지금까지 사용한 동전들의 집합이 state이고, 그 동전들을 나열하여 만든 합계가 amount일 때 가능한 순열(Permutation)의 수
- dp[state][amount] = Σ (dp[이전 state][이전 amount])
- 이전 state: state ^ (1 << i) (현재 집합에서 동전 i를 제외한 상태)
- 이전 amount: amount - coins[i] (현재 합계에서 동전 i의 값을 뺀 금액)
- 조건: 현재 state에 동전 i가 포함되어 있어야 함.
```python
# 중복 불가능한 순열 (Bitmask DP) - Hard
# n: 동전의 개수, target: 목표 금액
dp = [[0] * (target + 1) for _ in range(1 << n)]
dp[0][0] = 1

for state in range(1 << n):
    for amount in range(target + 1):
        if dp[state][amount] == 0: continue
        
        for i in range(n): # i번째 동전 선택 시도
            # 아직 i번째 동전을 사용하지 않았고, 합계가 target을 넘지 않는다면
            if not (state & (1 << i)) and amount + coins[i] <= target:
                next_state = state | (1 << i)
                dp[next_state][amount + coins[i]] += dp[state][amount]
```


### 7. Graph Traversal

**문제**: undirected 그래프에서 사이클 존재 여부 확인하기 (Cycle Detection)
```python
# DFS with Cycle Detection
def dfs(node, parent, visited, on_path):
    visited[node] = True
    on_path[node] = True
    
    for neighbor in graph[node]:
        if neighbor != parent and on_path[neighbor]:
            # neighbor가 원래 왔던 길이라면 cycle이 아님으로 무시한다. 만약 directed graph 라면 이 조건을 빼야한다.
            # neighbor가 이 DFS에서 이미 지나왔던 곳이라면 cycle이 생긴 것이므로 true 반환한다.
            return True
        elif not visited[neighbor]:
            if dfs(neighbor, node, visited, on_path):
                return True

    
    on_path[node] = False
    return False

# Initialize
visited = [False] * n  # 전체 작업에서 한 번이라도 방문했으면 true
on_path = [False] * n  # DFS 탐색 중에 방문한 것들, DFS 끝나면 원상복구
for i in range(n):
    if not visited[i]:
        if dfs(i, -1, visited, on_path):
            return True  # cycle found
return False  # no cycle
```

**문제**: 방향 그래프에서 선행 관계를 만족하는 작업 실행 순서 찾기 (Topological Sort)
```python
# Topological Sort (Kahn's Algorithm)
from collections import deque
in_degree = [0] * n
queue = deque()
result = []

# Calculate in-degrees
for u, v in edges:
    in_degree[v] += 1

# Start with nodes having 0 in-degree
for i in range(n):
    if in_degree[i] == 0:
        queue.append(i)

while queue:
    u = queue.popleft()
    result.append(u)
    for v in graph[u]:
        in_degree[v] -= 1
        if in_degree[v] == 0:
            queue.append(v)

return result if len(result) == n else []  # empty if cycle exists
```

### 8. Dutch National Flag
**문제**: 0, 1, 2 세 가지 값이 있는 배열 정렬하기 (Sort Colors)
```python
# Dutch National Flag Algorithm O(N)
low, mid, high = 0, 0, len(nums) - 1  
# [0, low-1]: 모두 0, [low, mid-1]: 모두 1, [mid, high]: 미지의 영역, [high+1, n-1]: 모두 2 
# 다음 0은 low로 와야한다. low 미만은 다 0이다.
# mid는 탐색 중인 포인터이다.
# 다음 2는 high 로 와야한다. high 초과는 다 2이다.
while mid <= high:
    if nums[mid] == 0:
        nums[low], nums[mid] = nums[mid], nums[low]
        low += 1
        mid += 1
    elif nums[mid] == 1:
        mid += 1
    else:  # nums[mid] == 2
        nums[mid], nums[high] = nums[high], nums[mid]
        high -= 1
# 색이 늘어나면 포인터를 추가해서 구간을 추가해주면 된다.
```

### 9. Binary Search

**문제**: 정렬된 배열에서 특정 조건을 만족하는 첫 번째 원소 찾기 (Binary Search Boundary)
```python
# Find first element >= target
left, right = 0, len(nums) - 1
result = -1
while left <= right:
    mid = left + (right - left) // 2
    if nums[mid] >= target:
        result = mid
        right = mid - 1
    else:
        left = mid + 1
return result
```

### 10. Dijkstra's Algorithm

**문제**: 양수 가중치 그래프에서 시작점에서 모든 노드까지의 최단 경로 찾기
- 시작점부터 각 도착점까지의 거리인 distance를 inf로 초기화한다.
- (dist from start, node) 를 heap 에 넣어서 heap 기반으로 BFS 처럼 탐색을 한다.
- heap pop 하고 인접 노드들을 확인한 뒤 인접 edge 만큼을 추가했을 때 기존 distance 보다 작아지면 업데이트하고 heap에 넣는다.
- Time: O(E log V), Space: O(V)
```python
import heapq
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]  # (distance, node)
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:  
            # 이 조건에 만족한다면 기존보다 더 오래 걸려서 온 거니까 무시하고 건너뛴다.
            continue
        for neighbor, weight in graph[node].items():
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                # new_dist 가 기존보다 크거나 같다면 무시하도록 한다.
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    return distances
```

### 11. Bellman-Ford Algorithm

**문제**: 음수 가중치 그래프에서 최단 경로 찾기 및 음수 사이클 감지
- 시작점부터 각 도착점까지의 거리인 distance를 inf로 초기화한다. distance[src] = 0 이다.
- edges 를 iterate 하면서 distance[start] 가 inf가 아니라면 해당 start node 에 도달이 가능하다는 것이므로 그 edge 를 계산해서 distance[end] 를 업데이트할 수 있으면 업데이트한다.
- 이러한 edge iteration을 (N-1) 만큼 반복한다. 그러면 src 부터 모든 노드까지 전파가 됨을 보장할 수 있고 최단 경로는 이 이내로 찾아져야한다.
- (N-1) 번 iterate 한 후 한 번 더 했을 때 또 업데이트되는 게 있다면 음수 사이클이 있다는 뜻이다.
- Time: O(VE), Space: O(V)
```python
def bellman_ford(edges, n, start):
    distances = [float('inf')] * n
    distances[start] = 0
    # Relax edges n-1 times
    # why n-1 times? 노드가 n 개인 그래프에서 사이클이 없는 최단 경로는 최대 n-1 개의 간선(edge)을 가질 수 있다.
    for _ in range(n - 1):
        for u, v, weight in edges:
            # 시작점이 도달 가능한 상태가 되어 있고 & 이 edge로 움직이는 게 도착점을 업데이트할 수 있다면
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
    # Check for negative cycles
    for u, v, weight in edges:
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            # n-1 번 iterate 했는데도 업데이트할 게 있다면 무한히 업데이트할 게 있다는 것이고 음수 사이클이 있다는 것이다.
            return None  # Negative cycle detected
    return distances
```

### 12. MST Algorithms

**문제**: 그래프의 모든 노드를 최소 비용으로 연결하는 최소 신장 트리 찾기 (Kruskal's Algorithm)
- edge를 weight 순서대로 정렬한다.
- 작은 weight의 edge 부터 확인하면서 그 edge가 양끝을 union 해준다면 선택한다. 만약 그 edge의 양끝이 이미 같은 set 이라면 무시한다.
- Time: O(E logE), Space: O(V+E)
```python
# Kruskal's Algorithm (Union-Find)
def kruskal(n, edges):
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            parent[px] = py
        elif rank[px] > rank[py]:
            parent[py] = px
        else:
            parent[py] = px
            rank[px] += 1
        return True
    
    edges.sort(key=lambda x: x[2])  # Sort by weight
    mst_weight = 0
    for u, v, weight in edges:
        if union(u, v):
            mst_weight += weight
    
    return mst_weight
```

**문제**: 그래프의 모든 노드를 최소 비용으로 연결하는 최소 신장 트리 찾기 (Prim's Algorithm)
- (edge의 목적지 노드, edge의 가중치) 를 저장하는 heap을 사용한다. 처음에 (0, start) 를 넣어놓는다.
- 처리가 된 노드는 저장하는 visited set 을 사용한다.
- heap pop 을 하면서 visited set 에 있는 노드라면 무시한다.
- 그렇지 않다면 해당 노드로는 지금 edge를 선택하는 게 최소의 weight로 추가하는 것이다.
  - why? 아직 탐색하지 않은 edge를 통해 가는 게 더 빠르다고 가정해보자. 그러면 그 탐색하지 않은 edge는 지금보다 더 느린 edge를 통해 찾아야한다. 그러는 순간 이미 더 비효율적인 탐색이 된 거싱다.
- 그 노드에 대해 neighbor를 탐색하여 visited 가 아닌 노드들을 추가한다.
- visited 의 크기가 전체 노드 수가 될 때까지 반복한다.
- Time: O(E logE), Space: O(E)
  - heap push, pop 을 모든 edge에 대해 수행해야할 수 있기 때문에 E x logE 가 된다.
```python
# Prim's Algorithm
import heapq
def prim(graph, start):
    visited = set()
    min_heap = [(0, start)]
    mst_weight = 0
    
    while min_heap and len(visited) < len(graph):
        weight, node = heapq.heappop(min_heap)
        if node in visited:
            continue
        visited.add(node)
        mst_weight += weight
        
        for neighbor, w in graph[node].items():
            if neighbor not in visited:
                heapq.heappush(min_heap, (w, neighbor))
    
    return mst_weight
```


### 13. BST Operations

**문제**: 이진 검색 트리에서 삽입, 삭제, 검색 연산 수행하기
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def insert(root, val):
    # 지금은 height를 맞추는 동작이 없어서 최악의 경우 O(N)이 걸릴 수 있다.
    # balanced tree로 만드려면 매번 height를 체크해서 rotate 하는 동작이 추가되어야 한다.
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert(root.left, val)
    else:
        root.right = insert(root.right, val)
    return root

def search(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return search(root.left, val)
    return search(root.right, val)

def delete(root, val):
    if not root:
        return root
    if val < root.val:
        root.left = delete(root.left, val)
    elif val > root.val:
        root.right = delete(root.right, val)
    else:
        if not root.left:
            return root.right
        if not root.right:
            return root.left
        # Find inorder successor
        min_node = root.right
        while min_node.left:
            min_node = min_node.left
        root.val = min_node.val
        root.right = delete(root.right, min_node.val)
    return root
```

### 14. Morris Traversal

**문제**: 이진 트리를 O(1) 공간으로 중위 순회하기
```python
def morris_inorder(root):
    current = root
    result = []
    
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                # left subtree의 rightmost 가 current의 바로 직전 값일 것이다. 
                # right child 가 없거나 right child 가 current가 된다면 탐색을 멈춘다.
                predecessor = predecessor.right
            
            if not predecessor.right:
                # right child 가 없다면 rightmost를 잘 찾아온 것이다.
                # rightmost 의 다음 값을 current로 연결해주고 다시 올라가서 탐색 시작한다.
                predecessor.right = current
                current = current.left
            else:
                # right child 가 current라면 한번 돌아서 올라온 것이다. 
                # current의 right subtree 처리가 다 끝났다는 뜻이므로 이제는 current를 처리해주면 된다.
                predecessor.right = None
                result.append(current.val)
                current = current.right
    
    return result
```

### 15. Trie Operations

**문제**: 문자열 삽입, 검색, 접두사 확인을 효율적으로 수행하기
```python
class TrieNode:
    def __init__(self):
        # TrieNode 자체에는 해당 객체가 어떤 char를 갖고 있는지 모르고 이 객체에 매핑된 key를 봐야한다. 
        # 필요하면 self.char 같은 걸 넣어야 하겠지만 지금은 필요가 없다.
        self.children = {}
        self.is_end = False  # 이 값이 True라면 root부터 현재 노드까지 내려오면서 거친 모든 문자들을 합친 단어가 있다는 뜻이다.

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

### 16. Bit Manipulation

**문제**: 비트 연산을 사용하여 효율적인 계산 수행하기
```python
# Count set bits
def count_set_bits(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# More efficient: Brian Kernighan's algorithm
def count_set_bits_efficient(n):
    count = 0
    while n:
        n &= (n - 1)  # Remove lowest set bit
        count += 1
    return count

# Check if power of 2
def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0

# Find single number in array where others appear twice
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

# Generate all subsets using bit manipulation
def subsets(nums):
    n = len(nums)
    result = []
    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)
    return result
```



### 17. Interval Problems

**문제**: 구간의 리스트가 주어졌을 때 겹치는 모든 구간을 병합하여 반환하기 (56. Merge Intervals)
```python
def merge_intervals(intervals):
    if not intervals: return []
    # 1. 시작 시간을 기준으로 정렬
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # 2. merged가 비었거나, 현재 구간의 시작이 이전 구간의 끝보다 크면(안 겹치면) 추가
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # 3. 겹치는 경우, 이전 구간의 끝값을 더 큰 값으로 갱신
            # 왜 max인가? [1, 10]과 [2, 6]이 올 수도 있기 때문
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged
```

**문제**: 구간들을 정렬하여 겹치는 영역이나 특정 지점의 상태 계산하기(Sweep Line Algorithm)
- 각 구간의 시작/끝점을 정렬하고 순회하며 현재 상태 업데이트
```python
# Sweep Line - Amount of New Area Painted Each Day
# 매일 일정 구간을 색칠하는데 새로 색칠하는 구간의 크기를 구하라.
def amount_painted_refined(paint):
    events = []
    for i, (s, e) in enumerate(paint):
        events.append((s, 1, i))  # 시작 이벤트 (타입 1)
        events.append((e, -1, i)) # 종료 이벤트 (타입 -1)
    events.sort()  # 위치순으로 정렬
    current_indices = set()  # 현재 색칠할 수 있는 날짜들의 모음이다.
    prev_pos = events[0][0]  # 이전 이벤트에서의 위치이다. current_indices는 prev_pos 부터 지금까지를 색칠할 수 있다는 뜻이다.
    res = [0] * len(paint)
    for pos, type, idx in events:
        # prev_pos 부터 pos 까지의 구간을 누가 칠할지는 결정된다. 
        # pos 지점을 칠해도 되나 헷갈렸는데 이건 점을 칠하는 게 아니라 구간을 칠하는 거니까 해도 된다. prev=1, pos=3 이라면 길이 2만큼 칠해야하는 것이다. 
        if current_indices:
            who_paints = min(current_indices)
            res[who_paints] += (pos - prev_pos)
        # 현재 이벤트를 바탕으로 상태 업데이트
        if type == 1:
            current_indices.add(idx)
        else:
            current_indices.remove(idx)   
        prev_pos = pos
    return res
```


### 18. Bidirectional Search

**문제**: 그래프에서 두 노드 사이의 최단 경로를 양방향에서 동시에 탐색하여 찾기
- 양쪽에서 BFS/DFS를 진행하며 탐색 영역이 만나는 지점에서 경로 합체
```python
# Bidirectional BFS for Shortest Path
def bidirectional_bfs(graph, start, end):
    if start == end: return 0
    # 딕셔너리에 거리 저장
    f_dist, b_dist = {start: 0}, {end: 0}
    f_queue, b_queue = deque([start]), deque([end])

    while f_queue and b_queue:
        # 항상 작은 큐를 f_queue가 되도록 스왑한다.
        # 양쪽에서 원을 확장시키면서 만나는 지점을 찾는 건데, 그 양쪽 원의 크기를 비슷하게 맞춰야 효율적이다.
        if len(f_queue) > len(b_queue):
            f_queue, b_queue = b_queue, f_queue
            f_dist, b_dist = b_dist, f_dist
        current = f_queue.popleft()
        for neighbor in graph[current]:
            if neighbor in b_dist:
                return f_dist[current] + 1 + b_dist[neighbor]
            if neighbor not in f_dist:
                f_dist[neighbor] = f_dist[current] + 1
                f_queue.append(neighbor)
    return -1
```


### 19. SPFA Algorithm (Shortest Path Faster Algorithm)

**문제**: 음수 가중치 그래프에서 Bellman-Ford를 큐로 최적화하여 최단 경로 찾기
- 각 노드가 큐에 들어간 횟수로 음수 사이클 감지, 업데이트된 노드만 큐에 재삽입
```python
# SPFA - Bellman-Ford Optimization
def spfa(edges, n, start):
    distances = [float('inf')] * n
    distances[start] = 0
    # 각 노드가 큐에 들어간 횟수 (음수 사이클 감지용)
    count = [0] * n
    queue = deque([start])
    in_queue = [False] * n
    in_queue[start] = True
    
    while queue:
        u = queue.popleft()
        in_queue[u] = False
        for v, weight in edges[u]:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                if not in_queue[v]:
                    queue.append(v)
                    in_queue[v] = True
                    count[v] += 1
                    # 음수 사이클 감지: 노드가 n번 이상 큐에 들어가면 사이클
                    if count[v] > n:
                        return None  # negative cycle detected
    
    return distances
```


### 20. Floyd-Warshall Algorithm

**문제**: 모든 정점 쌍 간의 최단 경로를 한 번에 계산하기
**점화식**: dp[k][i][j] = min(dp[k-1][i][j], dp[k-1][i][k] + dp[k-1][k][j])
```python
# Floyd-Warshall - All Pairs Shortest Path
def floyd_warshall(graph, n):
    # distance matrix 초기화
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0
            elif graph[i][j]:
                dist[i][j] = graph[i][j]
    # DP: k를 경유점으로 사용
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # i에서 j로 가는 것 vs i->k->j로 가는 것 비교
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist
```


### 21. LRU, LFC cache
LRU cache
- hashmap 으로 cache 의 key value를 관리하고, cache 내에서 사용하고 있는 key들의 사용 시점 관리는 double linked list 를 사용한다.
- cache size가 capacity를 넘어서면 linked list 맨 앞의 element가 가장 예전에 사용된 값이기 때문에 해당 값을 버려야한다. 
  - linked list 에서 해당 값 지우고, hashmap 에서도 del 명령어로 지운다. 각각 O(1) 이다.
- cache에 없는 key에 대해 get 요청이 들어오면, -1 반환한다.
- cache에 있는 key에 대해 get 요청이 들어오면, cache에서 바로 반환하고 해당 값을 linked list 맨 뒤에 추가해야한다. 기존에 linked list 에 있던 건 삭제해줘야한다. double linked list 이므로 앞뒤를 붙여주면 자동 삭제된다. 

```python
# LRU cache
class Node:
    def __init__(self, key, val):
        self.key, self.val = key, val
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cap = capacity
        self.cache = {}  # key: Node
        # Dummy Head와 Tail을 만들어두면 삽입/삭제가 편함
        self.head, self.tail = Node(0, 0), Node(0, 0)
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, node):
        """노드를 연결 리스트에서 제거"""
        p, n = node.prev, node.next
        p.next, n.prev = n, p

    def _add(self, node):
        """새 노드를 항상 tail 바로 앞에 삽입 (가장 최근 사용)"""
        p = self.tail.prev
        p.next = node
        self.tail.prev = node
        node.prev, node.next = p, self.tail

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)  # 사용했으므로 가장 최근 위치로 이동
            return node.val
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        
        new_node = Node(key, value)
        self._add(new_node)
        self.cache[key] = new_node
        
        if len(self.cache) > self.cap:
            # 가장 오래된 head.next 제거
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]
```

LFU cache
- Hash Map 2개 + Doubly Linked List 여러 개
  - cache: key를 넣으면 해당 데이터 노드를 반환한다.
  - freq_map: 빈도수를 넣으면 해당 빈도수를 가진 노드들의 Doubly Linked List를 반환한다.
  - min_freq: 현재 캐시 내 최소 빈도수를 추적하여 삭제 대상을 바로 찾는다.
- 로직
  - 어떤 키가 사용되면 빈도수를 +1 한다.
  - 해당 노드를 기존 freq_map[f] 리스트에서 빼서 freq_map[f+1] 리스트로 옮긴다.
  - 만약 기존 freq_map[f]가 비었고 그게 min_freq였다면, min_freq를 증가시킨다.


### 22. Top K Elements (Quick select)

문제: k번째 작은 원소를 O(N) 시간에 찾아라.
- 만약 Top K frequent element라면 각 원소마다 count를 세고, count -> element 를 갖는 bucket sort를 사용할 수도 있다. 
  - 하지만 공간 복잡도 적ㅇ로 비효율적이다. O(N)을 사용하게 된다.
  - 또한 count 기반이 아니라 value 기반이라면 O(maximum value) 가 되기 때문에 비효율적이다.
- quick select는 재귀 공간을 제외하고는 O(1)의 공간을 갖는다. 시간은 worst O(N^2)이 걸릴 수 있기 때문에 random pivot이 중요하다.
  - 매번 절반씩 탐색 구간을 줄인다면 N + N/2 + N/4 + ... = 2N = O(N)
  - 운이 나쁘게 매번 하나씩만 줄인다면 N + N-1 + N-2 + ... = O(N^2)
```python
def quick_select(nums, k):
    pivot = random.choice(nums)
    left = [x for x in nums if x < pivot]
    mid = [x for x in nums if x == pivot]
    right = [x for x in nums if x > pivot]
    
    if k <= len(left):
        return quick_select(left, k)
    elif k <= len(left) + len(mid):
        return pivot
    else:
        return quick_select(right, k - len(left) - len(mid))
```



## ⚡ 시간/공간 복잡도 최적화

### 공간 최적화 기법
1. **In-place 업데이트**: 입력 배열 직접 수정
2. **State Reduction**: 직전 상태만 저장
3. **Bit Mask**: 정수로 복수 상태 표현
4. **2D → 1D**: DP 테이블 압축

### 시간 최적화 기법
1. **Early Termination**: 불필요한 탐색 중단
2. **Memoization**: 중복 계산 제거
3. **Binary Search**: 정렬된 데이터에서 O(log N) 탐색
4. **Hash Lookup**: O(1) 평균 검색 시간

## Python 특화 팁

### 자주 쓰는 라이브러리
```python
from collections import defaultdict, deque, Counter
import heapq
from bisect import bisect_left, bisect_right
import math

# defaultdict: 키 없을 때 기본값
d = defaultdict(int)
d[key] += 1  # KeyError 없음

# Counter: 빈도수 계산
cnt = Counter(nums)
cnt.most_common(k)  # 상위 k개

# deque: 양방향 큐
q = deque([1, 2, 3])
q.append(4)      # 오른쪽 추가
q.appendleft(0)   # 왼쪽 추가
q.pop()          # 오른쪽 제거
q.popleft()      # 왼쪽 제거

# heap: 우선순위 큐
heap = [3, 1, 4, 1, 5]
heapq.heapify(heap)     # 리스트를 힙으로 변환
heapq.heappush(heap, 2) # 원소 추가
min_val = heapq.heappop(heap)  # 최솟값 제거

# bisect: 이진 탐색 삽입 위치
idx = bisect_left(sorted_list, value)  # 왼쪽 위치
idx = bisect_right(sorted_list, value) # 오른쪽 위치
```

### 주의사항
- **Shallow vs Deep Copy**: 
  ```python
  arr2 = arr[:]  # 1D에서는 안전
  arr2 = copy.deepcopy(arr)  # 2D+에서 필요
  ```
- **Variable Scope**: 중첩 함수에서 `nonlocal` 필요
- **String Immutability**: concatenation은 O(n) 비용
- **Integer Overflow**: Python에서는 걱정 없음

### 유용한 함수
```python
# 정렬
sorted_by_value = sorted(d.items(), key=lambda x: x[1])
min_key = min(d, key=d.get)

# 문자열
''.join(list_of_strings)  # 빠른 문자열 결합

# 수학
math.isqrt(n)  # 제곱근 정수 부분
math.gcd(a, b)  # 최대공약수
math.lcm(a, b)  # 최소공배수

# 리스트 컴프리헨션
squares = [x*x for x in range(10) if x % 2 == 0]
```

### 성능 최적화
```python
# 빠른 입출력
import sys
input = sys.stdin.readline

# 재귀 깊이 설정
sys.setrecursionlimit(10**6)

# 집합 연산 (O(1) 평균)
seen = set()
if x not in seen:  # O(1)
    seen.add(x)

# 딕셔너리 get 메소드
value = d.get(key, default_value)  # KeyError 방지
```

## 🔍 문제 해결 전략

### 1. 자료구조 선택
- **빠른 검색**: Hash Table O(1)
- **순서 필요**: Array, Linked List
- **최대/최소**: Heap O(log N)
- **범위 쿼리**: Prefix Sum, Segment Tree

### 2. 알고리즘 설계
- **Brute Force**: 먼저 생각해보기
- **최적화**: 중복 제거, 조기 종료, 더 좋은 자료구조
- **Edge Case**: 빈 입력, 단일 요소, 최대값

### 3. 구현 및 검증
- **예제 테스트**: 주어진 예시로 검증
- **Edge Case 테스트**: 경계값, 특수 케이스
- **복잡도 확인**: 요구사항 만족하는지


## 🚨 흔한 실수 & 해결책

### Floating Point Precision
```python
# ❌ 부동소수점 비교 => 컴퓨터에서 0.1 은 실제로는 0.1000.. 으로 정확히 0.1로 잘라지지 않는다.
if a == 0.1 + 0.2:

# ✅ epsilon 사용
epsilon = 1e-9
if abs(a - (0.1 + 0.2)) < epsilon:

# 또는 Decimal 사용
from decimal import Decimal
a = Decimal('0.1') + Decimal('0.2')
```

## 📊 복잡도 요약

| 자료구조/알고리즘      | 시간 복잡도     | 공간 복잡도        | 주요 사용처        |
|----------------|------------|---------------|---------------|
| BFS/DFS        | O(V+E)     | O(V)          | 그래프 탐색        |
| Dijkstra       | O(E log V) | O(V)          | 최단 경로 (양수)    |
| Bellman-Ford   | O(VE)      | O(V)          | 최단 경로 (음수)    |
| DP (1D)        | O(n)       | O(n) or O(1)  | 최적화 문제        |
| DP (2D)        | O(nm)      | O(nm) or O(m) | 격자, 문자열       |
| Union-Find     | O(α(n))    | O(n)          | 연결성, 사이클      |
| Kruskal(MST)   | O(E log E) | O(E)          | 최소 신장 트리      |
| Floyd-Warshall | O(V^3)     | O(V^2)        | 모든 노드 간 최소 거리 |
| Trie           | O(L)       | O(NL)         | 문자열 검색        |
| Heap           | O(log n)   | O(n)          | 우선순위 큐        |

* α(n): 역 애커만 함수, 거의 상수
* L: 문자열 길이
* V: 정점 수, E: 간선 수
