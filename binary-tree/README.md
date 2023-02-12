## 개념

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



## Binary Tree Traversal

- in-order traversal
inOrder(node.left) => visit(node) => inOrder(node.right)
작은 값부터 차례로 방문하게 된다.
- pre-order traversal
visit(node) => preOrder(node.left) => preOrder(node.right)
root부터 시작하는 traversal이다.
- post-order traversal
postOrder(node.left) => postOrder(node.right) => visit(node)
root가 제일 마지막에 visit된다.


recursive한 거는 iterative하게 구현할 수 있다. recursion도 call stack을 사용한다.



iterative한 in-order 탐색
```python
while stack or root:
  while root:
    stack.append(root)
    root = root.left
  root = stack.pop()
  # operation
  root = root.right
```

iterative한 pre-order 탐색

```python
while stack:
    curr_node = stack.pop()
    if curr_node:
        # operation
        stack.append(curr_node.right)
        stack.append(curr_node.left)

return answer
```

### Moris Traversal

