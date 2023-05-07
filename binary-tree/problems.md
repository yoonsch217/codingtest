
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


### 101. Symmetric Tree

https://leetcode.com/problems/symmetric-tree/description/

문제: binary tree의 root가 주어졌을 때 그 binary tree가 symmetric한지 구하라. 세로선을 기준으로 접었을 때 동일하게 겹쳐야 symmetric한 것이다.

level traversal 하는 방법으로 풀 수 있다.   
각 level의 list가 symmetric하면 다음 level로 넘어간다. symmetric하지 않으면 False return해야한다.   
이 때 주의할 점이 몇 가지 있다. 어떤 node의 child가 None이라도 next list에는 추가해줘야한다. 그래야 자리까지 정확히 symmetric한지 알 수 있다.   
그 next list를 갖고 다음 iteration을 할 때는 None인 node는 무시해도 된다. 이미 이전까지의 자리를 확인했기 때문에 유효한 node에 대해서만 추가로 진행하면 된다.   

<details>

```python
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    queue = deque([root])
    while queue:
        next_list = []
        while queue:
            cur = queue.popleft()
            if cur is None:
                continue
            next_list.append(cur.left)
            next_list.append(cur.right)
        n = len(next_list)
        for i in range(n//2):
            if next_list[i] is None and next_list[n-1-i] is None:
                continue
            if next_list[i] is None or next_list[n-1-i] is None or next_list[i].val != next_list[n-1-i].val:
                return False

        queue = deque(next_list)
    return True
```

</details>

역시 solution이 더 깔금하다...   
큐를 두고 처음에는 `[root, root]`를 넣는다. 처음 root는 왼쪽 subtree에 대한 root, 두 번째 root는 right subtree에 대한 root이다.   
그러고 두 개씩 pop을 하면서 left와 right를 동시에 비교한다. left subtree에 있는 노드에 대해서는 left, right 순서로 큐에 넣고, right subtree에 있는 노드는 반대 순서로 넣는다.   

<details>

```python
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    queue = deque([root, root])
    while queue:
        left = queue.popleft()
        right = queue.popleft()
        if left is None and right is None:
            continue
        if left is None or right is None:
            return False
        if left.val != right.val:
            return False
        queue.append(left.left)
        queue.append(right.right)
        queue.append(left.right)
        queue.append(right.left)
    return True
```

</details>


left와 right 각각을 위해 root를 두 번 넣는 것이 핵심이다.

비슷한 방식으로 recursive 하게도 할 수 있다. 

<details>

```python
def isSymmetric(self, root: Optional[TreeNode]) -> bool:
    def is_mirror(left_node, right_node):
        if left_node is None and right_node is None:
            return True
        if left_node is None or right_node is None:
            return False
        return left_node.val == right_node.val and is_mirror(left_node.left, right_node.right) and is_mirror(left_node.right, right_node.left)
    
    return is_mirror(root, root)
```

</details>



### 173. Binary Search Tree Iterator

https://leetcode.com/problems/binary-search-tree-iterator

문제: BSTIterator 라는 class는 binary search tree 를 받아서 construct 되는데 next 라는 함수와 hasNext 라는 함수를 갖는다. next는 호출될 때마다 가장 작은 수부터 차례대로 반환되고 hasNext는 다음 next가 존재하는지를 나타내는 boolean이다. next가 호출될 때는 항상 hasNext가 true인 상황이라고 가정을 한다. next 함수와 hasNext 함수를 구현하라.

나는 처음에 그냥 bst flatten을 해서 리스트를 만든 뒤에 앞에서부터 pointer를 옮겼다.    
이렇게 해도 되지만 solution은 stack을 사용했다.

<details>

```python
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        self._leftmost_inorder(root)

    def _leftmost_inorder(self, root):
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        topmost_node = self.stack.pop()
        if topmost_node.right:
            self._leftmost_inorder(topmost_node.right)
        return topmost_node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0
```

</details>


- 처음에 left child node만 다 stack에 추가한다. 
- 가장 작은 수는 top에 있을 것이고 next를 호출하면 그 값을 pop해야한다. 이제 top에 있는 수는 min의 parent 값이다.
- left child만 추가했었으므로 pop된 min의 right subtree는 고려가 안 됐다. min이 right subtree를 갖는다면 min의 parent보다 작은 값이 거기 있다.
- 따라서 pop된 min의 right subtree가 있는지 확인하고 있다면 그것도 동일하게 left child nodes들을 stack에 추가를 한다.

hasNext는 O(1)의 시간을 가질 것이다.   
next는 기본적으로 right subtree를 iterate하는 것이므로 O(N)의 시간이 필요하지만 amortized O(1)으로 볼 수 있다.   
space는 O(N)이 필요하다.





### 250. Count Univalue Subtrees

문제: root node가 주어졌을 때 uni value subtree의 수를 구하라. uni value subtree란 모든 노드의 값이 동일한 subtree를 말한다.


leaf node부터 확인을 한다. leaf node는 자기 자신 밖에 없으므로 항상 uni value subtree이다.   
그리고 어떤 node가 uni value subtree가 아니라면 그 모든 parent들도 uni value subtree가 아니다.   
어떤 node가 uni value subtree이려면 node.left와 node.right 모두 uni value subtree 이어야하고 node.val, node.left.val, node.right.val 이 모두 같아야한다.   

O(N) / O(N)

<details>


```python
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        self.count = 0

        def helper(cur):
            if not (cur.left or cur.right):
                self.count += 1
                return cur.val
            
            if cur.left and cur.right:
                from_left = helper(cur.left)
                from_right = helper(cur.right)
                if from_left == from_right and from_left == cur.val:
                    self.count += 1
                    return cur.val
                return None
            
            if cur.left:
                if helper(cur.left) == cur.val:
                    self.count += 1
                    return cur.val
                return None
            
            if cur.right:
                if helper(cur.right) == cur.val:
                    self.count += 1
                    return cur.val
                return None
        
        helper(root)
        return self.count
```


깔끔한 solution 풀이. return을 간단하게 했다. complexity는 똑같이 O(N) / O(N) 인데 이렇게 하니까 더 빨리 수행됐다. 이게 잘 짠 코드와의 차이이다..

```python
class Solution:
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        self.count = 0

        def dfs(node):
            if node is None:
                return True

            isLeftUniValue = dfs(node.left)
            isRightUniValue = dfs(node.right)

            if isLeftUniValue and isRightUniValue:
                if node.left and node.val != node.left.val:
                    return False
                if node.right and node.val != node.right.val:
                    return False
    
                self.count += 1
                return True
            return False
        
        dfs(root)
        return self.count
```

</details>

근데 global variable은 좋지 않은 코딩이다. 대신에 dfs 함수가 두 개의 값을 return하도록 한다.

<details>

```python
    def countUnivalSubtrees(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if node is None:
                return True, 0
            
            left = dfs(node.left)
            right = dfs(node.right)
            isLeftUniValue = left[0]
            isRightUniValue = right[0]
            count = left[1] + right[1]
            if isLeftUniValue and isRightUniValue:
                if node.left and node.val != node.left.val:
                    return False, count
                if node.right and node.val != node.right.val:
                    return False, count
                return True, count + 1
            return False, count
        
        return dfs(root)[1]
```

</details>





### 450. Delete Node in a BST

https://leetcode.com/problems/delete-node-in-a-bst/description/

문제: BST에서 root 가 주어지고 target이 주어졌을 때 target 값을 갖는 노드를 제거된 BST의 root를 반환하라.

어렵다. 이 함수는 target을 지운 뒤 그 tree의 root를 반환해준다.
target이 왼쪽에 있다면 deleteNode(target.left)의 결과는 leftsubtree에서 target을 지우고 그 subtree의 root이다.
그러면 root.left = deleteNode(target.left) 로 해주면 전체 tree에 대한 작업이 끝난다.

1. leaf node이면, 그 node를 None으로 바꾼다.
2. right child가 있으면 leftmost of the right subtree(successor)와 바꾼 뒤 그 node를 recursive하게 삭제한다.
3. left child가 있으면 rightmost of the left subtree(predecessor)와 바꾸고 그 node를 recursive하게 삭제한다.

헷갈리는 게, 어떤 객체 root에 대해서 root = None 하면 그 객체가 None이 돼?
root라는 변수가 None으로 reassign 되는 게 아니라?
root.parent.left = None 이런 식으로 해야하는 줄 알았다.
immutable
그럼 while successor.left: successor = successor.left 이런 식으로 하는 것도 포인터가 내려가는 게 아닌가?
포인터 개념인줄 알았는데.

<details>

solution

```python
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if root is None:
            return root
        
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not(root.left or root.right):
                root = None
            elif root.right:
                successor = root.right
                while successor.left:
                    successor = successor.left
                root.val = successor.val
                root.right = self.deleteNode(root.right, root.val)
            else:
                predecessor = root.left
                while predecessor.right:
                    predecessor = predecessor.right
                root.val = predecessor.val
                root.left = self.deleteNode(root.left, root.val)
        
        return root
```

</details>

user solution에 있던 더 간단한 코드.

1. target node의 left가 없으면 target node의 right를 반환하면 된다. right도 없다고 해도 괜찮다. 그럼 None이 반환되어야한다.
2. target node의 right가 없으면 target node의 left를 반환하면 된다.
3. target node가 left와 right 둘 다 있으면 successor를 찾아서 바꾼 뒤 successor node를 recursive하게 지워준다.

<details>

```python
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root: return None
        
        if root.val == key:
            if not root.right: return root.left
            
            if not root.left: return root.right
            
            if root.left and root.right:
                temp = root.right
                while temp.left: temp = temp.left
                root.val = temp.val
                root.right = self.deleteNode(root.right, root.val)

        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
            
        return root
```

</details>
