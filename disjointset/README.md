## 개념


### union & find

- 연결된 노드들끼리 grouping할 때 사용한다.
- 각 그룹마다 root node를 갖는다.
- root array라는 리스트가 있고 그 index는 노드의 id이다. 각 index의 값은 그 노드가 속한 그룹의 root 노드의 id이다.
- 처음에는 노드 리스트는 각 index의 값이 index가 들어가도록 한다.(자기 자신이 하나의 그룹이면서 root 노드이다.)



### Quick find implementation

find 작업을 빨리하는 구현이다.

- root array에 각 index의 값이 root index가 들어있다.
- (a, b) 와 같이 union할 때 b의 root를 찾아내고 root array를 쭉 돌면서 그 root의 자식들을 다 a의 root로 바꿔버린다.
- Time Complexity
  - contructor: O(N)
  - union: O(N) => root array 전체를 iterate해야하기 때문에
  - find: O(1) => 항상 root index가 저장된다.
  - connected: O(1)

<details>


```python
def find(self, x):
    return self.root[x]
    
def union(self, x, y):
    rootX = self.find(x)
    rootY = self.find(y)
    if rootX != rootY:
        for i in range(len(self.root)):
            if self.root[i] == rootY:
                self.root[i] = rootX
                
def connected(self, x, y):
    return self.find(x) == self.find(y)
```

</details>

### Quick union implementation

- union할 때 b의 root 값을 a의 root로 수정한다.
- Time Complexity
  - find: O(H), worst case O(N). 모든 원소가 일렬로 연결되어 있을 때
  - union: worst case O(N). 기본적으로 find를 한 후에 O(1)의 작업이 있기 때문에.
  - connected: worst case O(N). 마찬가지로 find가 필요하다.

일반적으로 quick union이 더 효율적이다.
왜냐하면 N개의 원소에 대해 작업을 할 때 quick find는 O(N^2)만큼 걸리지만 quick union은 worst case일 때 O(N^2)이 걸리기 때문이다.

<details>

```python
def find(self, x):
    while x != self.root[x]:
        x = self.root[x]
    return x
    
def union(self, x, y):
    rootX = self.find(x)
    rootY = self.find(y)
    if rootX != rootY:
        self.root[rootY] = rootX
```

</details>

### union by rank

- rank는 height와 비슷한 개념이다.
- rank가 낮은 vertices를 rank 높은 vertices 아래로 붙임으로써 height가 늘어나는 걸 막을 수 있다.
- union은 단순히 같은 그룹으로 묶기만 하면 되는 거다. 상하관계가 없는 것 같다.
- rank를 저장할 rank array를 하나 더 만들어야한다. rank에 그 root 아래에 몇 개가 있는지를 저장한다.
- Time Complexity
  - find: O(logN) 동일한 rank의 vertex끼리 계속 union되면 logN 높이의 그래프가 만들어진다.
  - union/connected: O(logN). 
  - find 작업에 dominant한 작업이다.

<details>

```python

def __init__(self, size):
    self.root = [i for i in range(size)]
    self.rank = [1] * size

def find(self, x):
    while x != self.root[x]:
        x = self.root[x]
    return x
    
def union(self, x, y):
    rootX = self.find(x)
    rootY = self.find(y)
    if rootX != rootY:
        if self.rank[rootX] > self.rank[rootY]:
            self.root[rootY] = rootX
        elif self.rank[rootX] < self.rank[rootY]:
            self.root[rootX] = rootY
        else:
            self.root[rootY] = rootX
            self.rank[rootX] += 1
```

</details>

### path compression

- find를 할 때 계속해서 타고 올라가야한다.
- 그렇게 올라가서 root를 찾은 후에 recursion으로 그 path의 vertex에 대해 root 를 수정해준다면 다음 find operation에서 효율성을 가질 수 있다.
- Time Complexity
  - find: best O(1), 
  - worst O(N), 
  - average O(logN)
  - union/connected: find dependent


### path compression + union by rank

- find/union이 O(1)으로 간주된다.
- construct는 O(N)

<details>

```python
def __init__(self, size):
    self.root = [i for i in range(size)]
    # Use a rank array to record the height of each vertex, i.e., the "rank" of each vertex.
    # The initial "rank" of each vertex is 1, because each of them is
    # a standalone vertex with no connection to other vertices.
    self.rank = [1] * size

# The find function here is the same as that in the disjoint set with path compression.
def find(self, x):
    if x == self.root[x]:
        return x
    self.root[x] = self.find(self.root[x])
    return self.root[x]

# The union function with union by rank
def union(self, x, y):
    rootX = self.find(x)
    rootY = self.find(y)
    if rootX != rootY:
        if self.rank[rootX] > self.rank[rootY]:
            self.root[rootY] = rootX
        elif self.rank[rootX] < self.rank[rootY]:
            self.root[rootX] = rootY
        else:
            self.root[rootY] = rootX
            self.rank[rootX] += 1

def connected(self, x, y):
    return self.find(x) == self.find(y)
```

</details>

### Check if all connected

```python
s = set()
for r in roots:
  s.add(find(r))
return len(s)
```






## 전략

하나의 그래프를 여러 개의 inter-connected graph로 나눠야 하는 graph partition 문제에 사용된다.


