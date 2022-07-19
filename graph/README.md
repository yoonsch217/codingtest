## 개념

그래프 안에 최대 pow(2, N-1) -1 개의 path가 있을 수 있다.   
기본적으로는 connected graph가 아니라면 모든 노드의 리스트를 저장하고 있어야한다.   
Tree의 경우는 root만 알면 다 조회가 가능하다.   

### 용어

- cut
그래프에서 두 개의 disjoint subset으로 나눌 때의 각 파티션이다.
- crossing edge
하나의 vertex set에서 다른 vertex set으로 연결하는 edge들이다.
- adjacency list
각 vertex는 인접한 노드의 리스트를 갖고 있다. undirected라면 연결된 두 노드는 서로의 노드를 저장한다.
- adjacency matrix
N개의 노드에 대해서 NxN matrix를 갖고 (i, j) 위치의 값은 i에서 j로의 edge가 있는지를 나타내는 boolean 값이다. undirected graph라면 symmetric matrix가 만들어진다.
edge 수가 node 수에 비해 상당히 클 때 효과적이다.
adjacent node를 iterate한다고 했을 때 adjacency list를 사용하면 바로 접근이 가능하지만 adjacency matrix를 사용하면 한 row를 다 읽어야한다.


### Tree

그래프의 일종으로, 여러 노드가 한 노드를 가리킬 수 없는 구조이다. => connected graph without a cycle   
- No cycle => unique path between two nodes
- Fully connected


### Binary Tree

child node가 최대 두 개인 트리이다. child가 없는 노드는 leaf라고 한다.   

- balanced binary tree
어떤 노드에서 보더라도 왼쪽 subtree와 오른쪽 subtree의 높이가 최대 1 차이나는 경우이다.
- complete binary tree
트리의 각 레벨이 다 채워져야하는데 마지막 레벨에서의 rightmost 노드는 비어있을 수 있다.
위에서 아래로, 왼쪽에서 오른쪽으로 차례대로 채워지는 트리이다.
이걸 array로 나타낸 경우 root가 index 1일 때, index n인 노드에 대해 parent node는 arr[n//2]이고 left child node는 arr[n * 2], right child node는 arr[n * 2 + 1]이다.
leaf node인지 아닌지는 index <= len(arr) // 2 이면 leaf node가 아닌 것이다.
- full binary tree
자식 노드가 0이거나 두 개인 트리이다. 한 개의 자식노드를 갖지 않는다.
- perfect binary tree
complete이면서 full인 binary tree이다. 모든 leaf 노드들이 같은 level에 있으며 꽉 차있다.


### Binary Search Tree(BST)

왼쪽 자식 노드들은 자기보다 다 작거나 같고, 오른쪽 자식 노드들은 다 자기보다 높다.   
in-order traverse를 하면 정렬된 순서로 방문한다.   
balanced 상태면 검색에 O(log N)이 걸리고 unbalanced면 최대 O(N)걸린다.    
balanced라는 조건이 없으면 root가 median이라는 보장이 없다.   

- insert
  - leaf에 넣는다. root부터 시작해서 자기 위치를 찾아 내려온 뒤 leaf 노드에서 알맞은 child leaf로 생성한다.
- delete
  - leaf라면 그냥 삭제한다. 
  - 하나의 child가 있는 경우 노드 삭제하고 child를 parent로 연결한다. 
  - child가 둘 이상인 경우 successor 노드를 찾아야한다. right subtree에서 가장 작은 노드를 찾아서 값을 바꾼 뒤 그 successor 노드를 삭제한다.



### Binary Tree Traversal

- in-order traversal
inOrder(node.left) => visit(node) => inOrder(node.right)
작은 값부터 차례로 방문하게 된다.
- pre-order traversal
visit(node) => preOrder(node.left) => preOrder(node.right)
root부터 시작하는 traversal이다.
- post-order traversal
postOrder(node.left) => postOrder(node.right) => visit(node)
root가 제일 마지막에 visit된다.


in-order 을 iterative하게 하는 법: stack을 둔다.
```python
while stack or root:
  while root:
    stack.append(root)
    root = root.left
  root = stack.pop()
  # operation
  root = root.right
```

### DFS

두 노드 사이의 path와 관련된 문제일 때 고려할 수 있는 방법이다.   
모든 path를 찾을 때 쓸 수 있다.   
stack 혹은 recursion을 사용한다.   
Time: O(V+E), Space: O(V)   

All possible paths를 구할 때 recursion을 사용하여 backtracking을 한다.   
이 때 각 작업마다 path를 고유하게 갖고 있어야하는데 처음에는 copy.deepcopy를 사용했었다.   
그런데 그렇게 하면 너무 시간이 많이 걸리고 dfs함수 부른 뒤 cur_path.pop()를 해주는 방식으로 시간을 절약할 수 있다.   
Time: O(2^N * N) => 가능한 path가 2^(N-1)-1, 각 path마다 다음 path 만들 때 O(N)개의 다음 node가 있으니까 O(N)의 시간 필요. 두 개 곱하면 loose upper bound   
Space: O(2^N * N) => 가능한 path가 2^(N-1)-1, 각 path마다 O(N) 노드가 있다.   

혹은 다른 개념이긴 하지만, dp(n): n에서 target까지의 모든 path list라고 할 때 dp(n) = [n] + next_path for next_path in dp(adj[n]) 로 dp로 풀수도 있다.   


### BFS

두 노드 사이의 shortest path 찾을 때 유용하다. 단 각 node, edge가 동일한 weight일 때 사용 가능하다.   
shortest path를 찾으려면 path를 리스트 복사해가면서 저장해야겠지만 shortest path length를 찾을 땐 count만 가져가면 된다.   
2d array에서 BFS의 space complexity: min(M, N)  https://imgur.com/gallery/M58OKvB   

### Bidirectional search

src에서 dest까지의 최단 경로를 찾는 방법이다..   
각 노드가 최대 k개의 adjacent node를 갖고, s에서 t까지의 최단 거리가 d라고 해보자.   
BFS를 사용하는 경우 exponential하게 증가하게 되고 O(k^d)의 시간이 걸린다.   

bidirectional search를 사용하면 s와 t 두 노드에서 동시에 시작하기 때문에 O(2 * k^d/2) 의 시간이 걸린다.   
s와 t에서 탐색을 하다가 두 그래프가 만나면 종료하게 된다.   
출발지와 목적지가 명확하게 주어지고, branching factor가 양방향에서 봤을 때 같을 때 사용하면 유용하다.   

양쪽에서 BFS를 하게 되는데 각 방향에 대해 queue와 parent라는 게 필요하다.   
parent는 그 탐색에서 어떤 노드에 대해 이전 노드를 저장하는 데이터이다.   
그래야 나중에 intersection을 기준으로 src까지의 path, dest까지의 path를 역으로 추적할 수가 있다.   


### Minimum spanning tree

spanning tree란 undirected graph에서 최소의 edge로 모든 vertex를 잇는 subgraph이다. 하나의 undirected graph는 여러 개의 spanning tree를 가질 수 있다.   

minimum spanning tree란 weighted undirected graph에서 최소의 edge weight을 갖는 spanning tree이다.   
cut property: 두 cut을 잇는 crossing edge 중 가장 weight가 작은 edge가 MST(minimum spanning tree)를 구성하게 된다.

- Kruskal's algorithm: edge를 추가하면서 MST를 만든다.
- Prim's algorithm: vertex를 추가하면서 MST를 만든다.


### Kruskal's Algorithm

1. edge를 weight가 증가하는 순서로 정렬한다.
2. weight가 작은 edge부터 MST에 추가한다. 이 때 cycle을 만드는 edge는 넘어간다.
3. N-1 edge를 찾을 때까지 2번 동작을 반복한다.

greedy algorithm이 적용된 방법이다.   
sorting을 해도 되고 heap을 써도 된다.   


Time Complexity: O(E log E)   
sorting하는 데 O(E log E) 시간이 걸리고 각 edge마다 양 끝 vertex가 같은 component에 연결되어있는지 확인하는 데 O(a(V)) 가 걸린다. a()는 Ackermann 함수이다. 따라서 O(E log E + E a(V)) = O(E log E) 이다.   
Space Complexity: O(V)   
union-find data structure를 사용하는 데 O(V)의 공간이 필요하다.   


### Prim's Algorithm

1. visited set, non-visited set 두 개를 둔다.
처음에는 visited set에 원소 하나를 둔다.
2. visited set 그룹에서 non-visited set 그룹으로 가는 crossing edge 중 가장 weight가 작은 걸 고른다.
visited 에 추가되는 각 element마다 edge 목록이 있는데 그걸 다 heap에 넣으면 항상 최소를 얻을 수 있다.
3. non-visited set 이 빌 때까지 반복한다.

greedy strategy를 사용한다.   

Time Complexity: O(E log V) for binary heap, O(E + V log V) for Fibonacci heap.   
Space Complexity: O(V)   



## 전략