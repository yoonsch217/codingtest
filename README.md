
### Basic Data Structures and Operations

- Array and String
  - Sorting
  - Sliding Window
  - Prefix Sum
  - Two Pointers
  - Sweep Line
  - Dutch National Flag
- Stack and Queue
  - Monotinic Stack
  - Monotonic Deque
  - Daily Temperatures: monotonic stack, backward iterate
  - Trapping Rain Water: left_maxs + right_maxs 로 구하기, monotonic stack
- HashTable
- Binary Search
  - o o o x x x => left: condition을 만족하지 않는 최소 index, right: 만족하는 최대 index
  - Find k closest elements: mid의 의미 정하기, subarray 크기를 하나 더 늘려서 어딜 버릴지 정하기
  - Find Minimum in Rotated Sorted Array: 리스트 데이터에 대한 이해


### Algorithmic Techniques

- Bit Manipulation
  - 기본 연산
    - `-64` original code, inverse code, complement code => 11000000 / 10111111 / 11000000
    - magnitude는 sign bit 빼고 전체 flip 후 1을 더하면 된다.
    - `&`, `|`, `^`, `<<`, `>>`
    - `~`: 각 bit flip. 양수에 대한 negation은 부호 바꾸고 magnitude를 하나 올린 값이다. (`~5` = -6)
    - Get i-th bit of the given number.: `num & (1<<i)`
    - Set i-th bit to 1: `num | (1<<i)`
    - Clear i-th bit: `num & (~(1<<i))`
    - `-a = ~(a-1)`
    - n & (n-1) 을 하게 되면 n에 존재하는 1 중에 가장 낮은 자리에 있는 1만 0으로 바꿔준다.
  - Bit Mask: 정수 하나로 리스트를 사용한 것과 같은 효과를 낸다. `10110(2) = 22`
  - State Compression via Bit Manipulation
- Recursion
  - backtracking: recursion을 들어가다가 invalid한 순간 멈추고 valid한 지점까지 돌아온다. DFS 생각하자.
- Greedy
  - Container with Most Water
- Dynamic Programming
  - State Reduction: 최근 n개만 필요하진 않은지, 상관관계가 있는 변수끼리 합칠 수 있는지 체크
  - Kadane's Algorithm: `i~j` 범위가 항상 커버되므로 valid한 알고리즘이다.
  - Knapsack algorithm
  - Best Time to Buy and Sell Stock with Cooldown: state가 여러 개 있고 각각의 관계를 구한 후 bottom up으로 업데이트한다.
  - Optimization on Space in Grid Problems: grid 자체를 업데이트하면 공간을 줄인다. 혹은 직전 row만 보관해도 가능한지 확인해본다.


### Grapshs and Trees

- Graph
  - all possible paths 수는 O(2^N)
  - BFS의 space O(min(M, N))
  - bidirectional search: src, dst 사이의 최단 경로 찾을 때 각각에서 BFS를 한다. `O(2*k^d/2) for d is the shortest distance`
  - Kruskal's Algorithm: MST. greedy + union-find. edge를 weight 순서로 정렬 후 작은 edge부터 cycle이 없다면 추가한다.
  - Prim's Algorithm: MST. greedy. visited set & edge heap. unvisited로 가는 가장 작은 edge부터 추가한다.
  - Dijkstra's Algorithm: greedy. `vtx_to_dist` dict과 `edge_heap(w, s, d)` heap이 필요하다.
  - Bellman-Ford Algorithm: dp. `dp[k][u]: 최대 k edge로 u 갈 때 최소 weight sum`
  - SPFA Algorithm: Bellman-Ford에서 iteration마다 연결된 edge만 큐에 넣어서 작업한다.
  - Kahn's Algorithm: topological sorting. `vtx_to_degree` dict
  - Cycle 확인하기: white, gray, black with DFS. gray 만나면 cycle.
  - Cheapest Flights within K Stops: DFS, Dijkstra, Bellman-Ford
- Binary Tree
  - BST insert: root부터 찾아서 알맞은 leaf에 추가한다, BST delete: right subtree의 leftmost leaf 노드와 swap 후 삭제한다.
  - Moris Traversal: O(N)/O(1) 탐색이다. cur와 last 두 포인터를 사용한다.
- Heap
- Trie
- Disjoint Set



### 풀어볼 문제들

https://leetcode.com/problems/unique-paths-iii/description/
https://leetcode.com/problems/concatenated-words/description/
https://leetcode.com/problems/parallel-courses-iii/description/
https://leetcode.com/problems/minimum-time-to-complete-all-tasks/description/
https://leetcode.com/problems/russian-doll-envelopes/description/
https://leetcode.com/problems/meeting-rooms-iii/description/
https://leetcode.com/problems/maximum-xor-with-an-element-from-array