# 코딩 인터뷰 Cheat Sheet

## 📋 기본 자료구조 및 알고리즘

### Array & String
- **Sliding Window**: O(N) 시간에 부분 배열/문자열 문제 해결
  - Two pointers로 window 크기 조절
  - 예: Longest Substring Without Repeating, Fruit Into Baskets
- **Two Pointers**: 정렬된 배열에서 효율적인 탐색
  - 예: 3Sum, Container With Most Water
- **Prefix Sum**: 구간 합 빠르게 계산 O(1)
- **Dutch National Flag**: 세 가지 값으로 분류 (Sort Colors)

### Stack & Queue
- **Monotonic Stack**: 감소/증가 순서 유지
  - Daily Temperatures: 뒤에서부터 탐색 O(N)
  - Trapping Rain Water: 웅덩이 층별 계산
- **Monotonic Queue**: Sliding Window Maximum
- **Min Stack**: 추가 스택으로 최솟값 관리 O(1)

### Hash Table
- **Anagram 그룹화**: 문자열 정렬 또는 문자 빈도수

### Binary Search
- **Pattern**: `oooxxx` 형태에서 경계 찾기
  - left: 조건 만족하지 않는 최소 index
  - right: 조건 만족하는 최대 index
- **응용**: K closest elements, rotated array 최솟값
- find_first 문제
  - 조건을 만족하는 경우 res에 임시 저장하고 다시 right = mid - 1 로 옮겨서 진행한다.

### Graph
- **BFS**: 최단 경로, 레벨 탐색 O(V+E)
- **DFS**: 사이클 감지, 위상 정렬
- **Dijkstra**: 최단 경로 (양수 가중치) O(E log V)
- **Bellman-Ford**: 음수 가중치, K stops 제한 O(VE)
- **Topological Sort**: Kahn's Algorithm (in-degree), DFS (white/gray/black)
- **MST**: Kruskal (Union-Find), Prim

### Tree
- **BST**: 삽입/삭제 O(log n)
- **Morris Traversal**: O(N)/O(1) 공간 순회
- **Trie**: 문자열 검색, 접두사

### Heap
- **우선순위 큐**: 최대/최솟값 빠른 접근 O(log n)
- **Top K Frequent**: Bucket sort O(N)

### Dynamic Programming
- **State Reduction**: 최근 n개만 필요한지 확인
- **Kadane's Algorithm**: 최대 부분합 O(N)
- **Knapsack**: 조합 문제
- **State Machine**: 여러 상태 간 전환 (Stock problems)

### Greedy
- **매 선택이 최적해로 이어짐**
- **Activity Selection, Fractional Knapsack**

### Bit Manipulation
- **기본 연산**: `&`, `|`, `^`, `<<`, `>>`, `~`
- **Bit Mask**: 정수로 상태 표현
- **n & (n-1)**: 가장 낮은 1 비트 제거

## 🎯 핵심 패턴 & 문제 유형

### 1. Subarray Problems
```python
# Maximum Subarray (Kadane)
max_sum = cur_sum = nums[0]
for num in nums[1:]:
    cur_sum = max(num, cur_sum + num)
    max_sum = max(max_sum, cur_sum)

# Sliding Window Fixed Size
from collections import deque
window = deque()
for i in range(len(nums)):
    window.append(nums[i])
    if i >= k-1:
        # process window
        window.popleft()
```

### 2. Two Pointers
```python
# Sorted Array Two Sum
left, right = 0, len(nums)-1
while left < right:
    s = nums[left] + nums[right]
    if s == target: return [left, right]
    elif s < target: left += 1
    else: right -= 1

# Sliding Window Variable Size
left = 0
for right in range(len(nums)):
    # expand window
    while condition_violated:
        # shrink window
        left += 1
```

### 3. Monotonic Stack
```python
stack = []  # stores indices
for i, val in enumerate(nums):
    while stack and nums[stack[-1]] < val:
        prev = stack.pop()
        # process popped element
        height = nums[prev]
        width = i if not stack else i - stack[-1] - 1
    stack.append(i)
```

### 4. BFS Shortest Path
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
```

### 5. DP Patterns
```python
# 1D DP - Fibonacci/Climbing Stairs
dp = [0] * (n+1)
dp[0] = dp[1] = 1
for i in range(2, n+1):
    dp[i] = dp[i-1] + dp[i-2]

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

# Knapsack (Combination)
dp = [0] * (target + 1)
dp[0] = 1
for coin in coins:
    for amount in range(coin, target + 1):
        dp[amount] += dp[amount - coin]
```

### 6. Graph Traversal
```python
# DFS with Cycle Detection
def dfs(node, parent, visited, recStack):
    visited[node] = True
    recStack[node] = True
    
    for neighbor in graph[node]:
        if not visited[neighbor]:
            if dfs(neighbor, node, visited, recStack):
                return True
        elif neighbor != parent and recStack[neighbor]:
            return True
    
    recStack[node] = False
    return False

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
from heapq import heappush, heappop, heapify
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
heapify(heap)     # 리스트를 힙으로 변환
heappush(heap, 2) # 원소 추가
min_val = heappop(heap)  # 최솟값 제거

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

| 자료구조/알고리즘 | 시간 복잡도 | 공간 복잡도 | 주요 사용처 |
|------------------|-------------|-------------|-----------|
| BFS/DFS | O(V+E) | O(V) | 그래프 탐색 |
| Dijkstra | O(E log V) | O(V) | 최단 경로 (양수) |
| Bellman-Ford | O(VE) | O(V) | 최단 경로 (음수) |
| DP (1D) | O(n) | O(n) or O(1) | 최적화 문제 |
| DP (2D) | O(nm) | O(nm) or O(m) | 격자, 문자열 |
| Union-Find | O(α(n)) | O(n) | 연결성, 사이클 |
| Trie | O(L) | O(NL) | 문자열 검색 |
| Heap | O(log n) | O(n) | 우선순위 큐 |

* α(n): 역 애커만 함수, 거의 상수
* L: 문자열 길이
* V: 정점 수, E: 간선 수
