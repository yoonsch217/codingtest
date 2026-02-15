
### 95. Unique Binary Search Trees II

https://leetcode.com/problems/unique-binary-search-trees-ii

문제: integer n이 주어진다. 1부터 n까지 n개의 노드를 갖고 만들 수 있는 unique BST 리스트를 순서 상관 없이 반환하라.

<details><summary>Approach 1</summary>

- root는 1부터 n까지 될 수 있다.
- k가 root가 되었을 때, left subtree는 1부터 k-1까지로 만들 수 있는 bst가 될 것이다.
- helper(start, end)를 start부터 end를 통해 만들 수 있는 bst 리스트라고 하자.
- 각 root k마다, helper(1, k-1) 중 하나가 left subtree가 되고 helper(k+1, n) 중 하나가 right subtree가 된다. 이걸 각각 엮으면 된다.

내 solution. 근데 helper(1,3) 이나 helper(2,4)나 동일한 답이고 value만 다른 건데 이 부분 최적화를 못 했다.    
length에 대해서 리스트 반환하는 함수를 만들고 그 함수 결과에 대해서 base만큼만 더해주면 더 빠를 것 같다.


```python
# class TreeNode:
#     def __init__(self, val:int =0, left: TreeNode =None, right:TreeNode=None):

def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
    @lru_cache(maxsize=None)
    def helper(start_idx: int, end_idx: int) -> List[TreeNode]:
        # Get all the bsts that can be created from the nodes from start_idx to end_idx
        if end_idx < start_idx:
            return [None]
        if start_idx == end_idx:
            return [TreeNode(start_idx)]
        rtn = []
        for cur in range(start_idx, end_idx+1):
            lss = helper(start_idx, cur-1)
            rss = helper(cur+1, end_idx)
            for ls in lss:
                for rs in rss:
                    rtn.append(TreeNode(cur, ls, rs))
        return rtn

    return helper(1, n)
```

</details>







### 366. Find Leaves of Binary Tree

https://leetcode.com/problems/find-leaves-of-binary-tree/ (locked)

문제: binary tree가 주어졌을 때, leaf 노드 리스트를 구하고, 그 leaf 노드를 없앤 tree의 leaf 노드 리스트를 구하면서 tree가 사라질 때까지의 leaf 노드 리스트를 순서대로 리스트에 넣은 2d array를 반환하라.

<details><summary>Approach 1</summary>

기본적으로 트리에는 height 0부터 max까지 있고 각각의 height에는 빈 리스트가 아님이 보장된다.    
traverse하면서 자기 height를 구하고 자기 위치에 맞는 곳에 들어간다. 맞는 위치에만 들어가면 되고 문제에서 순서는 중요하지 않다.

`height(root)=1+max(height(root.left), height(root.right))` 를 생각한다.   
그러려면 어떤 함수의 f(leftchild)와 f(rightchild) 값이 구해진 상태여야 현재 height를 구할 수 있기 때문에 left와 right 후에 자기 노드를 처리하는 post order를 사용한다.    
getHeight이라는 함수를 만드는데 이 함수는 post-order DFS를 하면서 leaf로부터의 height를 구한다.    
height를 구하면 `res[height].append(node.val)`을 해줌으로써 알맞은 위치에 답을 넣어주고 `return height`를 한다.    

O(N) / O(N)

</details>







### 2096. Step-By-Step Directions from a Binary Tree Node to Another

https://leetcode.com/problems/step-by-step-directions-from-a-binary-tree-node-to-another/

문제: 각 노드는 고유한 값을 갖는 binary tree의 root와 dest_value, start_value가 주어진다. 
start_value를 갖는 노드에서 dest_value를 갖는 노드로 가는 최단 경로를 찾아라. 
위로 가야한다면 U, 왼쪽 아래로 가야한다면 L, 오른쪽 아래로 가야한다면 R을 넣는다.

- Input: root = [5,1,2,3,null,6,4], startValue = 3, destValue = 6
- Output: "UURL"
- Explanation: The shortest path is: 3 → 1 → 5 → 2 → 6.



<details><summary>Approach 1</summary>

- 각각의 path를 구한 뒤 겹치는 부분을 제외한다.
- 공통 조상을 찾은 뒤, start 부터 공통조상 까지는 U 을 추가하고, 공통조상에서 dest 까지는 그 경로를 추가한다.

```py
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        def find(n: TreeNode, val: int, path: List[str]) -> bool:
            # find 함수 깔끔하다.
            if n.val == val:
                return True
            if n.left and find(n.left, val, path):
                path += "L"
            elif n.right and find(n.right, val, path):
                path += "R"
            return path

        s, d = [], []
        find(root, startValue, s)
        find(root, destValue, d)
        while len(s) and len(d) and s[-1] == d[-1]:
            s.pop()
            d.pop()
        return "".join("U" * len(s)) + "".join(reversed(d))
```

</details>








### 101. Symmetric Tree

https://leetcode.com/problems/symmetric-tree/description/

문제: binary tree의 root가 주어졌을 때 그 binary tree가 symmetric한지 구하라. 세로선을 기준으로 접었을 때 동일하게 겹쳐야 symmetric한 것이다.


<details><summary>Approach 1</summary>

level traversal 하는 방법으로 풀 수 있다.   
각 level의 list가 symmetric하면 다음 level로 넘어간다. symmetric하지 않으면 False return해야한다.   

이 때 어떤 node의 child가 None이라도 next list에는 추가해줘야한다. 그래야 자리까지 정확히 symmetric한지 알 수 있다.   
그 next list를 갖고 다음 iteration을 할 때는 None인 node는 무시해도 된다. 이미 이전까지의 자리를 확인했기 때문에 유효한 node에 대해서만 추가로 진행하면 된다.   

각 노드는 한 번씩만 방문하고, symmetric한지 체크하는 것도 각 리스트의 길이만큼만 필요하니까 O(2N) 이다.
O(N) / O(N)

또 다른 내 풀이로는, preorder traverse 하면서 list를 만든 뒤에 리스트를 비교했다.    
child node로 갈 때 left 먼저 갈지 right 먼저 갈지의 순서만 바꿨다. None도 넣어줘야한다.    
이것도 O(N) / O(N)의 복잡도이다.


</details>



<details><summary>Approach 2</summary>

역시 solution이 더 깔금하다...   

- 큐를 하나 두고 처음에는 `[left_root, right_root]`를 넣는다. left_root는 왼쪽 subtree에 대한 root, right_root는 right subtree에 대한 root이다.   
- 두 개씩 pop을 하면서 left와 right를 동시에 비교한다.  
- 두 개씩 빼면 한상 첫 번째 값은 left subtree, 두 번째 값은 right subtree 에서 온 게 보장이 된다.    
- 동일하다면 popped 노드에 대해 child node를 큐에 넣어줘야한다.   
- left subtree에 있는 노드에 대해서는 left, right 순서로 큐에 넣고, right subtree에 있는 노드는 반대 순서로 넣는다.   
- 이것도 전체 노드를 한 번씩 방문해야하고, 큐의 크기도 N이다. 그래도 전체 traverse할 필요가 없고, 각 level마다 가 아니라 각 node마다 검증하는 거니까 더 빨리 validate가 될 것 같다.
- O(N) / O(N) 


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




비슷한 방식으로 recursive 하게도 할 수 있다. 



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

문제: BSTIterator 라는 class는 binary search tree 의 root를 받아서 construct 된다. BSTIterator 내부에 next 라는 함수와 hasNext 라는 함수를 갖는다. 
next는 호출될 때마다 가장 작은 수부터 차례대로 반환되고 hasNext는 다음 next가 존재하는지를 나타내는 boolean이다. next가 호출될 때는 항상 hasNext가 true인 상황이라고 가정을 한다. next 함수와 hasNext 함수를 구현하라.

<details><summary>Approach 1</summary>

나는 처음에 그냥 bst flatten을 해서 리스트를 만든 뒤에 앞에서부터 pointer를 옮겼다.    
이렇게 해도 되지만 solution은 stack을 사용했다.

- 처음에 left child node만 다 stack에 추가한다. 
- 가장 작은 수는 top에 있을 것이고 next를 호출하면 그 값을 pop해야한다. 이제 top에 있는 수는 min의 parent 값이다.
- left child만 추가했었으므로 pop된 min의 right subtree는 고려가 안 됐다. min이 right subtree를 갖는다면 min의 parent보다 작은 값이 거기 있다.
- 따라서 pop된 min의 right subtree가 있는지 확인하고 있다면 그것도 동일하게 left child nodes들을 stack에 추가를 한다.

이것도 결국 in-order traversal과 동일하게 동작을 하는데 stack을 사용하는 방법이네. 대신 memory를 height 만큼까지 쓰니까 조금 더 효율적이다.



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

next: O(1) / O(h)

hasNext는 O(1)의 시간을 가질 것이다.   
next는 기본적으로 right subtree를 iterate하는 것이므로 O(N)의 시간이 필요하지만 amortized O(1)으로 볼 수 있다.    
space는 O(h)이 필요하다. 처음에 h만큼 넣고, _h만큼 올라갔다면, 그 right subtree에서 넣을 수 있는 데이터 양은 올라간 만큼인 _h이므로 계속 h 이하로 유지가 된다.

</details>








### 250. Count Univalue Subtrees

https://leetcode.com/problems/count-univalue-subtrees (locked)

문제: root node가 주어졌을 때 uni value subtree의 수를 구하라. uni value subtree란 모든 노드의 값이 동일한 subtree를 말한다.

<details><summary>Approach 1</summary>

leaf node부터 확인을 한다. 근데 subtree가 꼭 original tree의 leaf를 가져야하나? 지금은 문제가 lock돼서 안 보인다.

- leaf node는 자기 자신 밖에 없으므로 항상 uni value subtree이다.   
- 그리고 어떤 node가 uni value subtree가 아니라면 그 모든 parent들도 uni value subtree가 아니다.   
- 어떤 node가 uni value subtree이려면 node.left와 node.right 모두 uni value subtree 이어야하고 node.val, node.left.val, node.right.val 이 모두 같아야한다.   

O(N) / O(N)


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



근데 global variable은 좋지 않은 코딩이다. 대신에 dfs 함수가 두 개의 값을 return하도록 한다.



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

문제: unique value로 이루어진 BST에서 root 가 주어지고 target이 주어졌을 때 target 값을 갖는 노드가 제거된 BST의 root를 반환하라.

<details><summary>Approach 1</summary>

지우는 과정을 상상해서 단순화시킨 후 구현하기가 까다롭다.

이 함수는 target을 지운 뒤 그 tree의 root를 반환해준다.   
따라서 recursive하게 생각했을 때, target이 root 왼쪽에 있다면 `root.left = deleteNode(root.left, target)` 으로 하면 아래에서 알아서 된다.   


1. leaf node이면, 그 node를 None으로 바꾼다.
2. right child가 있으면 leftmost of the right subtree(i.e. successor)와 바꾼 뒤 그 node를 recursive하게 삭제한다.
3. left child가 있으면 rightmost of the left subtree(i.e. predecessor)와 바꾸고 그 node를 recursive하게 삭제한다.



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
                root = None  # 추가로 할 작업이 없다. 그냥 None을 반환하면 된다. 그럼 recursion에서 위에 있던 parent node에서 받아서 업데이트할 것이다.
            elif root.right:
                successor = root.right
                while successor.left:
                    successor = successor.left
                root.val = successor.val
                root.right = self.deleteNode(root.right, root.val)
            else:
                # 마찬가지로 predecessor를 찾아서 바꿔주면 되네. 나는 cur.parent를 찾아서 cur를 건너뛰고 연결하려고 하니까 복잡했다.
                predecessor = root.left
                while predecessor.right:
                    predecessor = predecessor.right
                root.val = predecessor.val
                root.left = self.deleteNode(root.left, root.val)
        
        return root
```

O(h) / O(h)


user solution에 있던 더 간단한 코드. `left가 있으면, right가 있으면` 의 조건이었는데 `left가 없으면, right가 없으면` 으로 바꾸니까 간단해진다.

1. target node의 left가 없으면 target node의 right를 반환하면 된다. right도 없다고 해도 괜찮다. 그럼 None이 반환된다.
2. target node의 right가 없으면 target node의 left를 반환하면 된다.
3. target node가 left와 right 둘 다 있으면 successor를 찾아서 바꾼 뒤 successor node를 recursive하게 지워준다.



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
