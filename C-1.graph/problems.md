README에 포함된 각 개념의 대표 예시 문제



### 743. Network Delay Time

https://leetcode.com/problems/network-delay-time

문제: n개의 노드가 주어지고 1번부터 n번까지 레이블이 되어 있다. times라는 리스트가 주어지는데 `times[i] = (u, v, w)` 로써 u 노드로부터 v 노드로 이동하는 데 w의 시간이 걸린다는 뜻이다. 노드 k에서 시작했을 때 모든 노드에 다 전파가 되는 데까지 걸리는 최소 시간을 구하라. 전파를 못 한다면 -1을 반환하라.


<details><summary>Solution</summary>

Dijkstra's Algorithm 문제이다. source는 k가 되고 k로부터 각각 노드까지의 최소 거리를 구한 후 그 max를 구하면 된다.

```py
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        # Dijkstra's Algorithm으로 source로부터의 최소 cost를 다 구하고 그것들의 max를 구한다.
        d = {}  # key: destination, value: distance, 만약 path가 필요하다면 (distance, previous) 이렇게 넣으면 된다.
        for i in range(n):
            d[i+1] = math.inf
        d[k] = 0

        adj_list = [[] for _ in range(n+1)]  # (dest, cost)
        for time in times:
            src, dest, cost = time
            adj_list[src].append((dest, cost))

        heap = [(0, k, k)]  # (distance, dest, src)
        
        while True:
            next_cost, next_vertex, prev_vertex = heapq.heappop(heap)
            for dest, cost in adj_list[next_vertex]:
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


<details><summary>Approach 1</summary>

BFS

queue를 사용해서 (node, price, stops)를 저장하면서 찾을 수 있다. stops가 k보다 작으면 k += 1 하면서 iterate한다.     
stops == k 인 경우는 k번을 다 쓴 거니까 그 때 목적지 노드가 dst라면 `res = min(res, price + cost_from_node_to_dst)` 로 업데이트하고 queue에는 데이터를 넣지 않는다.    
TLE 에러 발생.   

근데 optimize해서 O(EK) / O(V^2 + VK) 로 풀 수는 있다. (node, price, stops)를 넣는 게 아니라 k번 iterate하도록 한다.    
그리고 distances 라는 딕셔너리에 (node, stops) 로 stops만큼 갔을 때의 node까지 최소 거리를 저장한다.    
어떤 stop의 iteration에서 한 노드에 여러 edge가 접근할 수 있다. 그때 가장 최솟값을 구하기 위해서 (node, stops+1)에 대해 계속 참조하면서 업데이트해야한다.    


```python
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # todo: adj_matrix 만들기

        queue = deque()
        while queue and stops < k + 1:
            # Iterate on current level
            length = len(bfsQ)
            for _ in range(length):
                node = queue.popleft()
                
                # Loop over neighbors of popped node
                for nei in range(n):
                    if adj_matrix[node][nei] > 0:
                        dU = distances.get((node, stops), float("inf"))  # 이 값은 항상 있다. 이전 iteration에서 만든 값이다.
                        dV = distances.get((nei, stops + 1), float("inf"))  # 아직 unreached 상태라면 inf일테다. 있다면 다른 노드에서 간 결과인데 그거랑 비교하는 것이다.
                        wUV = adj_matrix[node][nei]
                        
                        if dU + wUV < dV:
                            distances[nei, stops + 1] = dU + wUV
                            queue.append(nei)
```


```py
# 내 BFS 솔루션: memory limit exceeded. 
# 노드들이 서로 다 연결되어 있다면 (worst case) 각 iteration마다 V개의 값이 queue에 들어갈 것이다. 이는 exponential하게 증가하므로 k^V의 space를 갖는다.
# visited 라는 set을 만들어서 하나의 path에서 중복된 vertex 방문을 막게 해봤는데 deepcopy를 사용하기 때문에 TLE가 난다.
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        adj_list = [[] for _ in range(n)]  # index: src, element: (dst, weight)
        for _src, _dst, _price in flights:
            adj_list[_src].append((_dst, _price))

        queue = deque([(src, 0, -1)])  # (current vertex, price sum, stops to get to this vertex)
        res = math.inf
        
        while queue:
            cur_vertex, price_sum, count = queue.popleft()
            for next_vertex, weight in adj_list[cur_vertex]:
                if next_vertex == dst:
                    res = min(res, price_sum + weight)
                    continue
                if count + 1 >= k:
                    continue
                queue.append((next_vertex, price_sum + weight, count+1))
            
        if res != math.inf:
            return res
        return -1
```

BFS를 할 때 모든 path를 다 탐색할 필요는 없다.    
dist라는 리스트를 만들어서 각 vertex까지 가는 최소의 total cost를 저장한다. 처음에는 inf로 초기화되어 있다.    
BFS가 늘어나면서 stops는 같거나 증가하게 되는데 total cost가 이전보다 늘어난다면 이후 path는 탐색하지 않아도 된다.   
해당 조건을 추가한 BFS로 하면 빠르다. Approach 2에 simple Dijkstra 가 이 방법이다.   
Further traversal을 하지 않아도 될 상황을 최대한 정교하게 생각하자.

</details>

<details><summary>Approach 2</summary>

Dijkstra's algorithm     

이 문제에서 주의해야할 점은 어떤 노드에 방문했을 때 전에 계산된 cost보다 더 높아도 지금의 count가 낮다면 둘 다 정답 후보가 된다는 것이다.    
예를 들어 노드 A에 5번 거쳐서 100의 price가 들었는데 이번 작업에서 2번 거쳐서 150의 price가 들었다고 하더라도 이 (A, 150, 2) 의 값도 유지해야한다.   
(A, 150, 2)가 (A, 100, 5)보다 더 낮은 가격으로 갈 순 없겠지만 최대로 도달할 수 있는 거리에서 차이가 난다.   
k가 5라면 (A, 100, 5)의 경우는 거기서 break를 해버리기 때문에 unreachable로 판단할 수 있다.   

`costs = [math.inf for _ in range(n)]`, `stops = [math.inf for _ in range(n)`으로 초기화시킨다. 각 vertex로 가는 cost/stops의 최솟값인가?    
minheap에 처음에 (cost, stops, node) = (0, 0, src) 를 넣는다.   
heappop을 하고 neighboring edges를 보면서 cost가 기존의 cost보다 작은지 확인한다.    
cost가 더 작다면 costs[node]와 stops[node] 를 업데이트한 후에 heap에 넣는다. cost가 작지 않더라도 stops가 더 작다면 heap에 넣는다.   
즉, cost가 더 작거나 stops가 더 작을 때 heap에 넣어주는 것이다.   

최소 cost가 되거나 최소 stops가 되면 candidate으로 보는 거 같은데, 적당히 작은 cost에 적당히 작은 stop이 candidate가 될 수도 있지 않을까? 정확히 구현은 안 해봐서 모르겠다. 
그렇다고 다 넣어버리면 너무 비효율적일 것 같고.

- Time Complexity: O(V^2logV)
- Space Complexity: O(V^2) adj_matrix



Simple Dijkstra => 이거 그냥 BFS 아냐?

- dist 값을 inf로 초기화하고 dist[src]만 0으로 한다. adj_list도 만들어놓는다.
- queue에 (stops, cur_node, price)를 넣는다. 초기에는 (0, src, 0)이 들어갈 것이다.
- while queue 조건동안 queue를 pop하면서 반복한다. stops가 k+1보다 크면 그 이후에 queue에 들어간 값은 다 k+1보다 크므로 break한다.
- cur_node에서 reachable한 node들의 dist를 확인해서 업데이트할 수 있는지 확인한다. `dist[reachable] > dist[cur_node] + price` 라면 거기로 갈 수 있는 거니까 업데이트하고 queue에 추가한다. (안 가는 거에 대한 건 저장을 안 해도 되나? 굳이 안 줄여도 될 수도 있잖아.)
- break가 되거나 queue가 비어서 loop를 빠져나오면 그 결과를 반환한다.



```python
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        adj_list = [[] for _ in range(n)]
        for _from, _to, _price in flights:
            adj_list[_from].append((_to, _price))
        dist = [math.inf] * n
        dist[src] = 0
        queue = deque([(0, src, 0)])  # stops, node, price

        while queue:
            stops, cur, price = queue.popleft()
            if stops > k:
                break
            for _to, _price in adj_list[cur]:
                if dist[_to] <= price + _price:
                    continue
                dist[_to] = price + _price
                queue.append((stops+1, _to, dist[_to]))

        return -1 if dist[dst] == math.inf else dist[dst]
```



</details>


<details><summary>Approach 3</summary>

Bellman Ford's Algorithm  

k번 iterate하면 k 번 hop했을 때의 결과를 알 수 있다.   
- dist list를 만들어서 source 빼고 inf로 초기화한다.   
- 전체 edge를 k+1번 iterate하면서 dist를 업데이트한다. 
- edge에서 from vertex의 distance가 inf라면 아직 시작을 못 하는 상태니까 넘어간다. from vertex가 inf가 아니면 여기에 k+1 이내로 도달할 수 있다는 것이므로 그 상태에서 to_index의 distance 값을 체크한다. 더 줄일 수 있으면 줄이고 없으면 무시한다.
- k+1번 iterate한 뒤에 dist[target] 값을 반환한다.

Bellman Ford 알고리즘은 기본적으로 dp이다. dp(k)를 구하기 위한 bottom up 방식이라고 생각하면 되겠다.    

```
dp(i): i번 iterate가 가능할 때 src에서 dst까지 가는 데 필요한 최소 비용
dp(i) = dp(i-1)의 상황에서 각 노드에서 한 번씩 주변을 업데이트 했을 때의 비용. 기준 노드에서 모든 edge를 탐색해서 기준노드의 cost + edge cost 값이 도착지의 cost보다 작다면 도착지의 cost를 업데이트한다.
```



```python
dist = [math.inf] * n
dist[src] = 0
for _ in range(k+1):
    next_dist = copy.deepcopy(dist)  # deepcopy를 해야한다. 안 그러면 edge iterate하면서 dist[_from] 값이 바뀔 수 있다.
    for _from, _to, _price in flights:
        if dist[_from] == math.inf:
            continue
        next_dist[_to] = min(next_dist[_to], dist[_from] + _price)  # 여기서 min의 first parameter는 dist가 아니라 next_dist가 되어야한다.
    dist = next_dist
if dist[dst] == math.inf:
    return -1
return dist[dst]
```
    

    
Time Complexity: O((V+E)*K), Space Complexity: O(V)

Bellman Ford는 각 iteration마다 한 번의 전체 이동을 하면서 dist를 업데이트하는 것이다. 그래서 iteration이 k+1번 일어난다.   
Dijkstra는 queue를 이용해서 stop이 점차 늘어난다.
    


</details>


<details><summary>Approach 4</summary>

SPFA 알고리즘    

Bellman Ford를 최적화시킨 알고리즘 => iteration마다 모든 edge를 탐색하는 게 아니라 영향을 줄 수 있는 edge만 저장해서 탐색한다.



```py
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        adj_list = [[] for _ in range(n)]  # index: from vertex, element: (to vertex, price)
        for _src, _dst, _price in flights:
            adj_list[_src].append((_dst, _price))

        distances = [math.inf for _ in range(n)]
        distances[src] = 0

        cur_list = [src]

        for _ in range(k+1):
            new_distances = copy.deepcopy(distances)
            next_list = []
            for cur in cur_list:
                for _dst, _price in adj_list[cur]:
                    if new_distances[_dst] <= distances[cur] + _price:
                        continue
                    new_distances[_dst] = distances[cur] + _price
                    next_list.append(_dst)
            distances = new_distances  # 여기 loop depth를 실수하지 말자.
            cur_list = next_list
        
        if distances[dst] == math.inf:
            return -1
        return distances[dst]
```

</details>









### 210. Course Schedule II

https://leetcode.com/problems/course-schedule-ii/

문제: numCourses와 prerequisites가 주어진다. prerequisites은 어떤 수업을 듣기 위해서 다른 수업을 들어야할 때가 있는데 그 정보가 있다. 들어야하는 수업 순서를 반환하라. 불가능하면 `[]`를 반환하라. `prerequisites[i] = [ai, bi]` indicates that bi must be taken before course ai.




<details><summary>Approach 1</summary>

Kahn's Algorithm

```py
    def findOrder(self, n: int, courses: List[List[int]]) -> List[int]:
        ordered_courses = []
        
        in_degrees = [0] * n  # in_degree[i] indicates the number of required courses to take course i
        adj_list = defaultdict(list)
        for next_course, required_course in courses:
            adj_list[required_course].append(next_course)
            in_degrees[next_course] += 1
        
        queue = deque()
        for i in range(n):
            if in_degrees[i] == 0:
                queue.append(i)
        
        while queue:
            cur = queue.popleft()
            ordered_courses.append(cur)

            for next_course in adj_list[cur]:
                in_degrees[next_course] -= 1
                if in_degrees[next_course] == 0:
                    queue.append(next_course)
        
        if len(ordered_courses) == n:
            return ordered_courses
        return []
```

Time: O(V+E), Space: O(V+E)이다.

</details>


<details><summary>Approcah 2</summary>

white, gray, black coloring을 이용한 DFS로도 풀 수 있다.

- adjacency list를 만들고 모든 vertex를 white 상태로 저장하고 시작한다.   
- 어떤 하나의 vertex부터 DFS를 한다.   
- DFS를 진행하면서 방문하는 vertex는 gray 상태로 바꾸면서 진행한다.   
- 어떤 gray vertex에서 더 이상 outbound가 없다면 그 course에 dependent한 게 없기 때문에 가장 나중에 들어야하는 course일 것이다. 그럼 그 vertex는 black으로 바꾸고 결과 stack에 추가한다.   
- 이후에 black vertex는 이미 처리가 됐으므로 없는 vertex라고 생각하면 된다. 즉, 어떤 gray vertex의 outbound가 black만 있다면 outbound가 없다고 생각하면 되므로 stack에 추가한다.   
- gray vertex의 outbound에 gray vertex가 있다면 cycle이 있는 것이기 때문에 순서를 정할 수가 없는 graph이다.   

Time: O(V+E), Space: O(V+E)이다.


</details>






---






README에 없는 문제들




### 1293. Shortest Path in a Grid with Obstacles Elimination

https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/

문제: grid라는 2d matrix가 주어지는데 각 값은 0 혹은 1이다. 0이 의미하는 건 그 위치에 장애물이 없다는 거고 1은 있다는 것이다. k번 장애물을 부수고 갈 수 있을 때 왼쪽 위에서 오른쪽 아래로 가는 최단 경로를 구하라. 가능한 경로가 없다면 -1을 반환하라.

<details><summary>Approach 1</summary>

BFS

최적화된 BFS가 필요하다. BFS에서 상태를 deepcopy 해서 넘기는 건 웬만하면 틀렸다고 생각하자.    

처음에는 queue를 두고 각 element는 (row, col, remained, visited) 을 넣었다.    
그러고는 매 iteration마다 모든 방향을 탐색한 뒤에 이동 가능하면 visited을 deepcopy하고 큐에 추가했다.    
(next_row, next_col) 이 목적지라면 len(visited)-1 만큼 이동한 것이다.    
그런데 이렇게 하면 동작은 하는데 TLE 제한에 걸린다.      

보완된 BFS     
loc_to_remained_and_steps 라는 dict를 정의해서 key는 (row, col), value는 (remained, steps)로 둔다.    
next location을 탐색할 때 `if (next_remained <= _remained and next_steps >= _steps) for any _remained, _steps in loc_to_remained_and_steps[(next_row, next_col)]` 이라면 이미 방문한 방법보다 무조건 비효율적일 수 밖에 없으므로 더 탐색을 하지 않아도 된다.    
즉, visited라는 상태를 복사해가는 대신 모든 case를 안고 가는데, 비효율적일 case는 버리고 가는 방식이다. 

그리고 k 값이 Manhattan distance보다 크다면 최단 거리로 갈 수 있으므로 그런 case를 처음에 처리하는 것도 도움이 된다. => 시간 확 줄었다.  

k 개의 다른 state를 가질 수 있기 때문에 각 노드마다 at most k번 방문한다. 
- Time Complexity: O(NK)
- Space Complexity: O(NK) 


약간 787. Cheapest Flights Within K Stops 의 simple Dijkstra 방식과 비슷해보인다? 상태를 deepcopy해가는 BFS가 실패한 것도 동일하고.   
787하고 다른 점은, 787의 경우는 stops 수만 고려하면 되어서 stops 수를 1부터 시작하여 bottom up을 할 수 있었다. 그래서 dp의 memo롤 1d로 만들 수 있었다.   
지금의 경우는 stops 수에 더해서 부술 수 있는 수까지 고려해야한다. 그러면 stops & remained 2d를 만들어야하는데 지금 위 solution이 이 방식인 것이었다.


```py
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        queue = deque([(0, 0, 0, k)])  # row, col, steps, remained
        
        if m == 1 and n == 1:
            return 0
        
        if k >= (m + n - 2):
            # if k is greater than or equal to Manhattan distance, return the minimum
            return m + n - 2

        """
        loc_to_remained_and_steps 변수
        key: (row, col), value: (remained, steps)
        탐색하다가 (next_row, next_col)이 loc_to_remained_and_steps 안에
          - 없으면 새로 넣는다.
          - 있고 지금 계산하는 상태가 갖고 있는 remained, steps가 loc_to_remained에 있는 어떤 데이터의 remained 보다 작고 steps도 많다면 더 비효율적으로 접근하는 것이기 때문에 멈춘다.
          - 아니라면 큐에 넣고 이후 작업을 계속 진행한다.
        """
        loc_to_remained_and_steps = defaultdict(list)
        loc_to_remained_and_steps[(0, 0)].append((k, 0))

        res = math.inf

        while queue:
            cur_row, cur_col, cur_steps, cur_remained = queue.popleft()

            for d_row, d_col in directions:
                next_row, next_col, next_remained = cur_row + d_row, cur_col + d_col, cur_remained

                if not (0 <= next_row < m and 0<= next_col < n):
                    # out of index
                    continue
                
                if grid[next_row][next_col] == 1:
                    # if encountered a block, reduce next_remained variable
                    next_remained = cur_remained - 1
                
                if next_remained < 0:
                    # cannot break block
                    continue
                
                if next_row == m-1 and next_col == n-1:
                    # arrived the target
                    res = min(res, cur_steps + 1)
                    continue
                
                is_visited = False
                for _remained, _steps in loc_to_remained_and_steps[(next_row, next_col)]:
                    if next_remained <= _remained and cur_steps+1 >= _steps:
                        # previously visited in a more effective way => do not have to do further operations
                        is_visited = True
                        break
                if is_visited:
                    continue
                    
                queue.append((next_row, next_col, cur_steps + 1, next_remained))
                loc_to_remained_and_steps[(next_row, next_col)].append((next_remained, cur_steps+1))

        if res == math.inf:
            return -1
        return res
```

A* algorithm도 있다는데 이건 우선 skip

</details>










### 1059.All Paths from Source Lead to Destination

https://leetcode.com/problems/all-paths-from-source-lead-to-destination

문제: edges 라는 directed graph가 주어진다. edges[i] = [ai, bi] 는 ai에서 bi로 가는 edge가 있다는 걸 의미한다. source와 destination이 주어졌을 때 source에서 시작되는 모든 path는 destination으로 가는지를 구하라.


<details><summary>Approach 1</summary>

내 처음 solution => TLE    
DFS를 사용하면 모든 path를 탐색할 수 있다. 하나의 일련의 recursion은 path를 만들게 된다.   
recursion을 하면서 visited 변수를 사용해서 다음 노드가 visited 안에 있다면 이건 destination으로 못 가고 cycle이 생긴다는 뜻이므로 return false 한다.   
recursion을 하면서 현재의 노드에서 더 이상 갈 곳이 없다면 현재 노드가 destination인지 확인하고 return true/false 한다.   



dfs를 사용했다.

```py
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        adj_list = defaultdict(list)
        for _s, _d in edges:
            adj_list[_s].append(_d)
        
        if len(adj_list[source]) == 0:
            return source == destination
        
        def helper(cur, visited):
            if len(adj_list[cur]) == 0:
                return cur == destination

            visited.add(cur)
            for next_node in adj_list[cur]:
                if next_node in visited:
                    return False
                if not helper(next_node, visited):
                    return False  # 여기서 early return을 안 하고 all(helper_results) 로 하면 속도가 느려진다.
            visited.remove(cur)
            return True

        return helper(source, set())  # 또 set 하는 것 보다는 is_visited dict를 만들어서 각 vertex마다 true/false를 갖게 하는 게 낫겠다.
```

TLE 실패

</details>


<details><summary>Approach 2</summary>

동일한 컨셉인데 visited set의 add/remove 대신 coloring을 사용했다.   
DFS를 하면서 white, gray, black 세 가지의 색으로 상태를 칠한다.    

- 처음에 모든 노드는 white 상태이다.
- DFS를 하면서 지금 지나는 노드는 gray로 칠한다.
- 어떤 노드에서 edge가 없다면 destination인지 아닌지 체크를 한다. destination이라면 true, 아니면 false를 반환한다.
- 어떤 노드에서 자기의 edge들에 대해 탐색이 모두 true로 끝나면 자기 노드를 black으로 칠하고 return true한다. 아래 recursion에서도 black으로 칠하고 올라왔을 것이다.
- 탐색을 하다가 black을 만나면 true를 반환한다. 그 노드에서는 이미 모두 destination으로 갔다는 걸 알기 때문이다. => 일종의 subproblem 느낌도 있네.
- 탐색을 하다가 gray를 만나면 false를 반환한다. cycle이 생겼다는 뜻이기 때문이다.



```py

class Solution:
    GRAY = 1
    BLACK = 2

    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        graph = self.buildDigraph(n, edges)
        return self.leadsToDest(graph, source, destination, [None] * n)
        
    def leadsToDest(self, graph, node, dest, states):
        
        # If the state is GRAY, this is a backward edge and hence, it creates a Loop.
        if states[node] != None:
            return states[node] == Solution.BLACK
        
        # If this is a leaf node, it should be equal to the destination.
        if len(graph[node]) == 0:
            return node == dest
        
        # Now, we are processing this node. So we mark it as GRAY.
        states[node] = Solution.GRAY
        
        for next_node in graph[node]:
            
            # If we get a `false` from any recursive call on the neighbors, we short circuit and return from there.
            if not self.leadsToDest(graph, next_node, dest, states):
                return False
        
        # Recursive processing done for the node. We mark it BLACK.
        states[node] = Solution.BLACK
        return True
        
    def buildDigraph(self, n, edges):
        graph = [[] for _ in range(n)]
        
        for edge in edges:
            graph[edge[0]].append(edge[1])
            
        return graph   
```

time: O(V), space: O(V+E)

일반적인 DFS는 O(V+E)의 시간 복잡도를 갖는다. 모든 vertex끼리 연결되어 있는 경우 E는 (V^2)이다.    
하지만 이 solution에서는 cycle detection을 pruning and backtracking 로 하기 때문에 O(V)에 할 수 있다.

</details>


