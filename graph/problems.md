### 210. Course Schedule II

https://leetcode.com/problems/course-schedule-ii/

문제: numCourses와 prerequisites가 주어진다. prerequisites은 어떤 수업을 듣기 위해서 다른 수업을 들어야할 때가 있는데 그 정보가 있다. 들어야하는 수업 순서를 반환하라. 불가능하면 `[]`를 반환하라.

Topological sort 문제이다. 먼저 defaultdict을 사용해서 adjacent list를 생성한다.    
그리고 각 노드마다 state를 둔다. WHITE는 방문한 적 없는, DFS를 시작할 노드이다. GREY는 방문한 적 있지만 아직 마지막까지 안 간 노드이다. BLACK은 그 노드에 대한 모든 작업을 마친 노드이다.   
모든 노드에 대해 iterate하면서 WHITE 상태이면 DFS를 한다. 그리고 DFS하면서 그 노드가 BLACK이 되면 topologically sorted array에 차례대로 추가를 한다.   
그 topologically sorted array를 뒤집은 게 정답이다.   
그리고 cycle이 있으면 불가능해진다. 따라서 DFS를 하다가 GREY를 다시 만난다면 cycle이 있는 것이므로 실패를 반환한다.


### 95. Unique Binary Search Trees II

https://leetcode.com/problems/unique-binary-search-trees-ii

문제: integer n이 주어진다. 1부터 n까지 n개의 노드를 갖고 만들 수 있는 unique BST를 순서 상관 없이 반환하라.

일반화를 잘 시켜야한다.    
BST인 경우, 특정 노드를 기준으로 왼쪽 subtree에는 그 노드보다 작은 모든 값이 있어야 하고, 오른쪽 subtree에는 그 노드보다 큰 모든 값이 있어야한다.   
따라서 left, right가 주어졌을 때 그 사이의 각 값들이 root이고 left~root-1 이 left subtree, root+1~right 가 right subtree가 되도록 만든다.   
즉, 하나의 root에 대해 helper(left, root-1) 가 left subtree로 가능한 root 목록이고 helper(root+1, right)가 right subtree로 가능한 목록이니까 이걸 2 depth loop로 엮어준다.


### 366. Find Leaves of Binary Tree

https://leetcode.com/problems/find-leaves-of-binary-tree/

문제: binary tree가 주어졌을 때, leaf 노드 리스트를 구하고, 그 leaf 노드를 없앤 tree의 leaft 노드 리스트를 구하면서 tree가 사라질 때까지의 leaf 노드 리스트를 순서대로 리스트에 넣은 2d array를 반환하라.


기본적으로 `height(root)=1+max(height(root.left), height(root.right))` 를 생각한다.   
getHeight이라는 함수를 만드는데 이 함수는 post-order DFS를 하면서 leaf로부터의 height를 구한다.    
height를 구하면 `res[height].append(node.val)`을 해줌으로써 알맞은 위치에 답을 넣어주고 `return height`를 한다.   
O(N) / O(N)


### 1293. Shortest Path in a Grid with Obstacles Elimination

https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/

문제: grid라는 2d matrix가 주어지는데 각 값은 0 혹은 1이다. 0이 의미하는 건 그 위치에 장애물이 없다는 거고 1은 있다는 것이다. k번 장애물을 부수고 갈 수 있을 때 왼쪽 위에서 오른쪽 아래로 가는 최단 경로를 구하라. 가능한 경로가 없다면 -1을 반환하라.

처음엔 BFS로 했다. queue를 두고 각 node는 (row, col, remained, seen) 을 넣었다. 그러고는 매 iteration마다 모든 방향을 탐색한 뒤에 이동 가능하면 seen을 deepcopy하고 큐에 추가했다.    
그런데 이렇게 하면 동작은 하는데 TLE 제한에 걸린다.      

이 방법은 seen을 처리하는 데 비효율작이다. 매 iteration마다 deepcopy하는 데는 많은 비용이 든다.     
모든 iteration이 공통으로 쓸 수 있는 seen을 생각해보면 seen에 (row, col, remained)라는 state를 넣어주는 방법이 있다. 대신 queue에 넣을 때 state와 steps까지 넣어줘야한다.        
이렇게 하면 deepcopy의 비용도 줄일 수 있고, 서로 다른 iteration에서 같은 위치를 방문할 때 이전에 이미 동일한 state(row, col, remained)로 방문했다면 이번 iteration에서 방문하는 게 더 짧을 수 없기 때문에 불필요한 path 생성을 막아준다.    
O(N*K) / O(N*K)  => 각 노드마다 at most k번 방문한다. k 개의 다른 state를 가질 수 있기 때문에.

그리고 k 값이 Manhattan distance보다 크다면 최단 거리로 갈 수 있으므로 그런 case를 처음에 처리하는 것도 도움이 된다.  

A* algorithm도 있다는데 이건 우선 skip


### 2096. Step-By-Step Directions From a Binary Tree Node to Another

https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/

문제: binary tree가 주어지고 각 node는 고유한 값을 갖는다. start_value와 dest_value가 있을 때 start_value를 갖는 노드에서 dest_value를 갖는 노드로 가는 최단 경로를 찾아라. 위로 가야한다면 U, 왼쪽 아래로 가야한다면 L, 오른쪽 아래로 가야한다면 R을 넣는다.

처음에 생각했던 건, start node를 찾아서 BFS로 dest node를 찾는 것이었다. parent node를 모르기 때문에 start node찾을 때 각 노드의 parent 노드를 저장해야한다. 그런데 이건 많이 비효율적이다.   
대신에 root에서 DFS로 각각의 노드를 찾는다. 찾게 되면 recursion 탈출하면서 경로를 만들 수 있다.   
각각에 가는 path를 찾게 되면 common prefix를 찾아서 없앤다. 그 뒤에 start path는 다 U로 바꾸고 dest path를 추가해주면 된다.   
DFS는 이렇게 구현했다: base case로는 node가 없으면 return None, target node이면 []를 반환하도록 했다. 그리고 left child와 right child를 각각 recursive 호출하면서 None이면 무시, 아니면 어디로 갔는지를 기록해준다. 이 때 처음에 틀린 거는, 이렇게 append하면 root에서의 path는 거꾸로 봐야한다는 점을 놓쳤었다.


### 743. Network Delay Time

https://leetcode.com/problems/network-delay-time

문제: n개의 노드가 주어지고 1번부터 n번까지 레이블이 되어 있다. times라는 리스트가 주어지는데 `times[i] = (u, v, w)` 로써 u 노드로부터 v 노드로 이동하는 데 w의 시간이 걸린다는 뜻이다. 노드 k에서 시작했을 때 모든 노드에 다 전파가 되는 데까지 걸리는 최소 시간을 구하라. 전파를 못 한다면 -1을 반환하라.

Dijkstra's Algorithm 문제이다. source는 k가 되고 k로부터 각각 노드까지의 최소 거리를 구한 후 그 max를 구하면 된다.

<details>

```python
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Dijkstra's Algorithm으로 source로부터의 최소 cost를 다 구하고 그것들의 max를 구한다.
        d = {}  # value: distance, 만약 path가 필요하다면 (distance, previous) 이렇게 넣으면 된다.
        for i in range(n):
            d[i+1] = math.inf
        d[k] = 0

        adj_matrix = [[] for _ in range(n+1)]  # (dest, cost)
        for time in times:
            src, dest, cost = time
            adj_matrix[src].append((dest, cost))

        heap = [(0, k, k)]  # (distance, next, previous)
        
        while True:
            next_cost, next_vertex, prev_vertex = heapq.heappop(heap)
            for dest, cost in adj_matrix[next_vertex]:
                if d[dest] <= d[next_vertex] + cost:
                    continue
                d[dest] = d[next_vertex] + cost
                heapq.heappush(heap, (cost, dest, next_vertex))
            if len(heap) == 0:
                break

        res = max(map(lambda x: d[x], d))
        return res if res != math.inf else -1
```

</details>


### 787. Cheapest Flights Within K Stops

https://leetcode.com/problems/cheapest-flights-within-k-stops/description/
    
문제: n개의 노드가 있고 flights라는 리스트가 있다. flights에는 (출발지, 목적지, 가격) 의 데이터가 리스트로 저장되어 있다. 최대로 들를 수 있는 노드의 수가 k로 주어졌을 때 src 노드에서 dst 노드로 가는 최소 요금을 구하라. 만약 갈 수 없다면 -1을 반환하라.

이 문제에서 주의해야할 점은 어떤 노드에 방문했을 때 원래 있던 price가 더 낮더라도 지금의 count가 낮다면 지금의 값을 버릴 수 없다는 것이다.    
예를 들어 노드 A에 5번 거쳐서 100의 price가 들었는데 이번 작업에서 2번 거쳐서 150의 price가 들었다고 하더라도 이 (A, 150, 2) 의 값도 유지해야한다.   
(A, 150, 2)가 (A, 100, 5)보다 더 낮은 가격으로 갈 순 없겠지만 최대로 도달할 수 있는 거리에서 차이가 난다.   
k가 5라면 (A, 100, 5)의 경우는 거기서 break를 해버리기 때문에 unreachable로 판단할 수 있다.   

Approach 1: BFS   
queue를 사용해서 (node, price, stops)를 저장하면서 찾을 수 있다. stops == k 인 경우는 continue하고 아닌 경우는 k += 1 하면서 iterate한다.     
만약 목적지 노드가 dst라면 `res = min(res, price+edge['price'])` 로 업데이트하면 된다.
TLE 에러 발생.   
근데 optimize해서 O(EK) / O(V^2 + VK) 로 풀 수는 있다. (node, price, stops)를 넣는 게 아니라 k번 iterate하도록 한다. 
그리고 distances 라는 딕셔너리에 (node, stops) 로 stops만큼 갔을 때의 node까지 최소 거리를 저장한다.    
어떤 stop의 iteration에서 한 노드에 여러 edge가 접근할 수 있다. 그때 가장 최솟값을 구하기 위해서 (node, stops+1)에 대해 계속 참조하면서 업데이트해야한다.

<details>

```python
        while bfsQ and stops < K + 1:
            
            # Iterate on current level
            length = len(bfsQ)
            for _ in range(length):
                node = bfsQ.popleft()
                
                # Loop over neighbors of popped node
                for nei in range(n):
                    if adj_matrix[node][nei] > 0:
                        dU = distances.get((node, stops), float("inf"))  # 이 값은 항상 있다. 이전 iteration에서 만든 값이다.
                        dV = distances.get((nei, stops + 1), float("inf"))  # 아직 unreached 상태라면 inf일테다. 있다면 다른 노드에서 간 결과인데 그거랑 비교하는 것이다.
                        wUV = adj_matrix[node][nei]
                        
                        if dU + wUV < dV:
                            distances[nei, stops + 1] = dU + wUV
                            bfsQ.append(nei)
```

</details>


Approach 2: Dikjstra's algorithm
기존의 알고리즘하고의 차이점은 cost가 작지 않더라도 cnt가 작으면 저장해야한다는 것이다.   
`costs = [math.inf for _ in range(n)]`, `stops = [math.inf for _ in range(n)`으로 초기화시키고 minheap에 처음에 (cost, stops, node) = (0, 0, src) 를 넣는다.   
heappop을 하고 neighboring edges를 보면서 cost가 기존의 cost보다 작은지 확인한다. cost가 더 작다면 costs[node]와 stops[node] 를 업데이트한 후에 heap에 넣는다. cost가 작지 않더라도 stops가 더 작다면 heap에 넣는다.   
즉, cost가 더 작거나 stops가 더 작을 때 heap에 넣어주는 것이다.   
Time Complexity: O(V^2logV), Space Complexity: O(V^2) adj_matrix

Approach 3: Bellman Ford's Algorithm   
k번 iterate하면 k 번 hop했을 때의 결과를 알 수 있다.   
- dist list를 만들어서 source 빼고 inf로 초기화한다.   
- 전체 edge를 k+1번 iterate하면서 dist를 업데이트한다. 
- source가 inf라면 아직 시작을 못 하는 상태니까 무시한다. source가 inf가 아니면 여기에 k+1 이내로 도달할 수 있다는 것이므로 그 상태에서 dest의 dist 값을 업데이트한다. 더 줄일 수 있으면 줄이고 없으면 무시한다.
- k+1번 iterate한 뒤에 dist[target] 값을 반환한다.

<details>

```python
dist = [math.inf] * n
dist[src] = 0
for _ in range(K+1):
    next_dist = copy.deepcopy(dist)
    for _from, _to, _price in flights:
        if dist[_from] == math.inf:
            continue
        next_dist[_to] = min(next_dist[_to], dist[_from] + _price)
    dist = next_dist
if dist[dst] == math.inf:
    return -1
return dist[dst]
```
    
</details>
    
Time Complexity: O((V+E)*K), Space Complexity: O(V)

Bellman Ford는 각 iteration마다 한 번의 전체 이동을 하면서 dist를 업데이트하는 것이다. 그래서 iteration이 k+1번 일어난다.   
Dijkstra는 queue를 이용해서 stop이 점차 늘어난다.
    
Simple Dijkstra
- dist 값을 inf로 초기화하고 dist[src]만 0으로 한다. adj_list도 만들어놓는다.
- queue에 (stops, cur_node, price)를 넣는다. 초기에는 (0, src, 0)이 들어갈 것이다.
- while queue 조건동안 queue를 pop하면서 반복한다. stops가 k+1보다 크면 그 이후에 queue에 들어간 값은 다 k+1보다 크므로 break한다.
- cur_node에서 reachable한 node들의 dist를 확인해서 업데이트할 수 있는지 확인한다. `dist[reachable] > dist[cur_node] + price` 라면 거기로 갈 수 있는 거니까 업데이트하고 queue에 추가한다. (안 가는 거에 대한 건 저장을 안 해도 되나? 굳이 안 줄여도 될 수도 있잖아.)
- break가 되거나 queue가 비어서 loop를 빠져나오면 그 결과를 반환한다.

<details>

```python
dist = [math.inf] * n
dist[src] = 0
queue = deque([(0, src, 0)])  # stops, node, price

while queue:
    stops, cur, price = queue.popleft()
    if stops > K:
        break
    for _to, _price in adj_list[cur]:
        if dist[_to] <= price + _price:
            continue
        dist[_to] = price + _price
        queue.append((stops+1, _to, dist[_to]))

return -1 if dist[dst] == math.inf else dist[dst]
```

</details>
