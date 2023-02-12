
### 95. Unique Binary Search Trees II

https://leetcode.com/problems/unique-binary-search-trees-ii

문제: integer n이 주어진다. 1부터 n까지 n개의 노드를 갖고 만들 수 있는 unique BST를 순서 상관 없이 반환하라.

일반화를 잘 시켜야한다.    
BST인 경우, 특정 노드를 기준으로 왼쪽 subtree에는 그 노드보다 작은 모든 값이 있어야 하고, 오른쪽 subtree에는 그 노드보다 큰 모든 값이 있어야한다.   
따라서 left, right가 주어졌을 때 그 사이의 각 값들이 root이고 left ~ root-1 이 left subtree, root+1 ~ right 가 right subtree가 되도록 만든다.   
즉, 하나의 root에 대해 helper(left, root-1) 가 left subtree로 가능한 root 목록이고 helper(root+1, right)가 right subtree로 가능한 목록이니까 이걸 2 depth loop로 엮어준다.


### 366. Find Leaves of Binary Tree

https://leetcode.com/problems/find-leaves-of-binary-tree/

문제: binary tree가 주어졌을 때, leaf 노드 리스트를 구하고, 그 leaf 노드를 없앤 tree의 leaft 노드 리스트를 구하면서 tree가 사라질 때까지의 leaf 노드 리스트를 순서대로 리스트에 넣은 2d array를 반환하라.


기본적으로 `height(root)=1+max(height(root.left), height(root.right))` 를 생각한다.   
getHeight이라는 함수를 만드는데 이 함수는 post-order DFS를 하면서 leaf로부터의 height를 구한다.    
height를 구하면 `res[height].append(node.val)`을 해줌으로써 알맞은 위치에 답을 넣어주고 `return height`를 한다.   
O(N) / O(N)




### 2096. Step-By-Step Directions From a Binary Tree Node to Another

https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/

문제: binary tree가 주어지고 각 node는 고유한 값을 갖는다. start_value와 dest_value가 있을 때 start_value를 갖는 노드에서 dest_value를 갖는 노드로 가는 최단 경로를 찾아라. 위로 가야한다면 U, 왼쪽 아래로 가야한다면 L, 오른쪽 아래로 가야한다면 R을 넣는다.

처음에 생각했던 건, start node를 찾아서 BFS로 dest node를 찾는 것이었다. parent node를 모르기 때문에 start node찾을 때 각 노드의 parent 노드를 저장해야한다. 그런데 이건 많이 비효율적이다.   
대신에 root에서 DFS로 각각의 노드를 찾는다. 찾게 되면 recursion 탈출하면서 경로를 만들 수 있다.   
각각에 가는 path를 찾게 되면 common prefix를 찾아서 없앤다. 그 뒤에 start path는 다 U로 바꾸고 dest path를 추가해주면 된다.   
DFS는 이렇게 구현했다: base case로는 node가 없으면 return None, target node이면 []를 반환하도록 했다. 그리고 left child와 right child를 각각 recursive 호출하면서 None이면 무시, 아니면 어디로 갔는지를 기록해준다. 이 때 처음에 틀린 거는, 이렇게 append하면 root에서의 path는 거꾸로 봐야한다는 점을 놓쳤었다.
