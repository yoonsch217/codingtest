### 1202. Smallest String With Swaps

https://leetcode.com/problems/smallest-string-with-swaps

문제: string s가 주어지고 index pair로 이루어진 리스트 pairs가 주어진다. 리스트의 각 원소인 index pair에 있는 두 index는 각각 s의 index를 의미하고 그 두 index끼리는 swap이 가능하다. 횟수 제한 없이 swap을 했을 때 lexicographically smallest string을 반환하라.

pairs에 union-find를 한다. 그런데 서로 연결된 index끼리 자유롭게 swap이 가능하다는 걸 어떻게 증명하는지 몰랐다. 실험적으로만 알았고.   
[0, 1], [1, 2] 이렇게 있으면 [0, 1, 2]가 자유롭게 교환 가능하다는 걸.   
swap을 두 번 하면 제자리로 온다. 그럼 한 번 하고, 다른 character를 원하는 위치로 옮긴 후 다시 swap을 하면 제자리로 돌아올 것이다. 이 원리인가.   
- union find로 root array를 만들어서 grouping한다.
- key: root, value: reachable character list 로 dict를 만든 뒤 각 list를 정렬한다.
- string 길이만큼의 인덱스를 앞에서부터 돌면서 자기 위치의 그룹에 있는 character를 추가한다.
- 자기 위치 그룹의 character list에서 어디까지 추가했는지를 기록하기 위해 key: root, value: count 의 dict도 필요하다.
- Time complexity: O((E+V)⋅α(V)+VlogV)

<details>

```python
for x, y in pairs:
    union(x, y)

d = defaultdict(list)  # key: root, value: reachable character list
for i, c in enumerate(s):
    d[find(i)].append(c)

for key in d:
    d[key].sort()

res = []
d_cnt = defaultdict(int)
for i in range(len(s)):
    root_i = find(i)
    res.append(d[root_i][d_cnt[root_i]])
    d_cnt[root_i] += 1

return ''.join(res)
```

</details>


DFS로도 가능하다.



### 399. Evaluate Division

https://leetcode.com/problems/evaluate-division/description/

문제: equations 와 values 라는 리스트가 있는데 `equation[i][0] / equation[i][1] = values[i]` 를 만족한다. equation[i] 는 string 두 개로 이루어진 리스트이고 values는 숫자로 이루어진 리스트이다. (즉 관계식을 나타낸다고 볼 수 있다.)
또한 queries 라는 리스트가 주어지는데 이 리스트도 각각의 element가 string 두 개로 이루어진 리스트이다. 
이 queries의 각 element에 대해서 `output[i] = queries[i][0] / queries[i][1]` 을 만족하는 output 리스트를 반환하라. 
구할 수 없는 게 있으면 -1.0 을 넣는다.


solution에 있는 union find 기법이다.   
- root array 대신 var_to_gid_weight 라는 dictionary를 사용한다. key는 equation에 사용되는 변수값, value는 (gid, gid 기준의 multiple) 의 tuple 을 갖는다.
- find는 var를 받고 (gid, multiple) 값을 반환한다. 기본 find 동작과 로직은 동일한데 path compression을 하면서 이전 gid 기준 weight에서 새로운 gid 기준 weight를 업데이트해줘야한다.
- union은 dividend, divisor, result 세 값을 받아야한다. 두 변수의 gid가 같다면 추가적으로 할 작업은 없다. gid가 다르다면 divisor의 gid 기준으로 dividend의 var_to_gid_weight를 업데이트해준다.
- 결과 구할 때는 변수가 var_to_gid_weight에 둘 다 포함이 안 되면 -1을 넣는다. 포함되는 게 있으면 gid를 각각 구해서 다르면 -1을 넣는다. gid가 같다면 계산을 한다.


<details>

```python
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:

        gid_weight = {}

        def find(node_id):
            if node_id not in gid_weight:
                gid_weight[node_id] = (node_id, 1)
            group_id, node_weight = gid_weight[node_id]
            # The above statements are equivalent to the following one
            #group_id, node_weight = gid_weight.setdefault(node_id, (node_id, 1))

            if group_id != node_id:
                # found inconsistency, trigger chain update
                new_group_id, group_weight = find(group_id)
                gid_weight[node_id] = (new_group_id, node_weight * group_weight)
            return gid_weight[node_id]

        def union(dividend, divisor, value):
            dividend_gid, dividend_weight = find(dividend)
            divisor_gid, divisor_weight = find(divisor)
            if dividend_gid != divisor_gid:
                # merge the two groups together,
                # by attaching the dividend group to the one of divisor
                gid_weight[dividend_gid] = (divisor_gid, divisor_weight * value / dividend_weight)

        # Step 1). build the union groups
        for (dividend, divisor), value in zip(equations, values):
            union(dividend, divisor, value)

        results = []
        # Step 2). run the evaluation, with "lazy" updates in find() function
        for (dividend, divisor) in queries:
            if dividend not in gid_weight or divisor not in gid_weight:
                # case 1). at least one variable did not appear before
                results.append(-1.0)
            else:
                dividend_gid, dividend_weight = find(dividend)
                divisor_gid, divisor_weight = find(divisor)
                if dividend_gid != divisor_gid:
                    # case 2). the variables do not belong to the same chain/group
                    results.append(-1.0)
                else:
                    # case 3). there is a chain/path between the variables
                    results.append(dividend_weight / divisor_weight)
        return results
```

O((M+N) * logN) / O(N)

</details>





근데 이거를 그래프로 이해할 수도 있다. 예를 들어 a/b = 2 인 경우 a에서 b로 가는 edge가 있고 그 weight가 2라고 표현할 수도 있는 것이다.   
화살표 방향이 나누는 방향으로 사용할 수 있기 때문에 directed graph를 쓴다.   
a/c 를 구할 때는 a에서 c로 가는 path를 DFS로 찾으면서 곱셈 혹은 나눗셈을 하면 된다.      
graph를 구할 때 보통은 노드 index가 있어서 matrix로 구하는데 여기서는 그렇게 할 수가 없다.   
그런 경우는 `graph = defaultdict(defaultdict)` 으로 해서 그냥 `graph[start][end] = weight` 으로 넣어버린다.   


<details>

```python
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        
        graph = defaultdict(defaultdict)

        def backtrack_evaluate(curr_node, target_node, acc_product, visited):
            visited.add(curr_node)
            ret = -1.0
            neighbors = graph[curr_node]
            if target_node in neighbors:
                ret = acc_product * neighbors[target_node]  # 목적지에 다다렀으니까 더 탐색을 안 해도 된다.
            else:
                for neighbor, value in neighbors.items():
                    if neighbor in visited:  # 이미 들렀던 곳이니까 못 간다.
                        continue
                    ret = backtrack_evaluate(
                        neighbor, target_node, acc_product * value, visited)
                    if ret != -1.0:
                        break
            return ret

        # Step 1). build the graph from the equations
        for (dividend, divisor), value in zip(equations, values):
            # add nodes and two edges into the graph
            graph[dividend][divisor] = value
            graph[divisor][dividend] = 1 / value

        # Step 2). Evaluate each query via backtracking (DFS)
        #  by verifying if there exists a path from dividend to divisor
        results = []
        for dividend, divisor in queries:
            if dividend not in graph or divisor not in graph:
                # case 1): either node does not exist
                ret = -1.0
            elif dividend == divisor:
                # case 2): origin and destination are the same node
                ret = 1.0
            else:
                visited = set()
                ret = backtrack_evaluate(dividend, divisor, 1, visited)
            results.append(ret)

        return results
```

O(MN) / O(N)

</details>






### 952. Largest Component Size by Common Factor

https://leetcode.com/problems/largest-component-size-by-common-factor/


문제: nums 라는 int 리스트가 주어지고 각 element는 하나의 노드이다. 1이 아닌 공약수를 가지면 서로 undirectedly 연결이 된다. 가장 많이 연결된 수를 subgraph의 노드 수를 구하라.

처음에 각 num마다의 공약수 set을 만들고 has_common_factor 함수를 `len(s1 & s2) > 0` 조건으로 하려고 했는데 TLE 난다. intersection operation은 꽤 긴 시간이 걸린다.   

- 각 숫자마다 prime set을 구한다. 예를 들어 12라면 (2, 3)이 prime set이다. 
- prime_to_num dictionary를 만들어서 같은 prime을 갖는 num 끼리 union을 해준다.
- union 된 그룹 중 가장 큰 그룹의 크기를 반환한다.

prime set 구하는 것도 까다로웠다. 이 접근 방식은 외워둬야할 것 같다.

<details>

```py
class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        node_to_root = {}
        def find(target):
            if target not in node_to_root:
                node_to_root[target] = target
            if target == node_to_root[target]:
                return target
            
            node_to_root[target] = find(node_to_root[target])
            return node_to_root[target]
        
        def union(target1, target2):
            root1 = find(target1)
            root2 = find(target2)
            node_to_root[root2] = root1

        # Time Complexity: O(N * sqrt(m))
        def get_prime_set(num):
            if num == 1:
                return set()
            for i in range(2, int(sqrt(num)) + 1):
                if num % i == 0:
                    return get_prime_set(num//i) | set([i])
            return set([num])
        

        prime_to_num = defaultdict(list)
        # Time complexity: O(N * log(m)). At most log(m) prime divisors can exist
        for num in nums:
            prime_set = get_prime_set(num)
            for pr in prime_set:
                prime_to_num[pr].append(num)
        
        for pr in prime_to_num:
            cur_nums = prime_to_num[pr]
            for i in range(len(cur_nums)-1):
                union(cur_nums[i], cur_nums[i+1])
        
        root_to_nodes = defaultdict(list)
        for node in node_to_root:
            root_to_nodes[find(node)].append(node)
        
        cnt = 1  # initialize this to "1"
        for root in root_to_nodes:
            cnt = max(cnt, len(root_to_nodes[root]))
        
        return cnt
```

time complexity는 get_prime_set 함수가 정한다.   
nums 길이를 N, 가장 큰 num을 m이라고 하면 `O(N * sqrt(m))` 이 시간복잡도가 된다.



</details>

