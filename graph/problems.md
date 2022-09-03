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




