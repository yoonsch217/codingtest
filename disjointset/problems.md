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

즉, 방정식에서 관계식을 통해서 곱셈 나눗셈을 하는 문제이다. 그리고 어떤 두 미지수 사이의 관계식을 구하지 못하면 곱셈 혹은 나눗셈이 실수로 떨어지지 않는데 그럴 때는 -1.0을 넣으라는 뜻이다.

처음에 좀 복잡했는데 그냥 brute force하게 모든 미지수 사이의 관계식을 모두 구하지! 로 시작했다가 disjoint set 의 아이디어를 적용해서 조금 더 간단하게 했다.    
근데 사실 disjoint set을 그대로 사용한 건 아니고 아이디어만 빌린 것 같다.   
rank 대신에 len of set 으로 어딜 어디로 붙일지 정하고, union 할 때 전체 iterate하면서 다 통일해주고 하는 등의 아이디어를 빌려왔다.

각 미지수를 나타낼 때 보통 x = 2*y 이런 식으로 하는데 이런 표현이 파이썬에서는 그대로 사용할 수가 없다.   
그래서 각 미지수마다 x = 2*y 라면 `string_to_value[x] = (y, 2)` 이렇게 관계식을 저장한 dictionary를 정의했다.

<details>

my solution

```python
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        string_to_value = {}  # key: string, value: (base string, repetition)
        base_to_strings = defaultdict(set)  # key: base string, value: strings that are represented based on the base string

        def update(s1, s2, res):
            # "s1 = s2 * res" => "s2 = s1 / res"
            s1_base, s1_base_rep = string_to_value[s1]
            if s2 in string_to_value:
                # "s1_base * s1_rep = s2_base * s2_rep * res" => "s2_base * 1 = s1_base * (s1_rep / s2_rep / res)"
                s2_base, s2_base_rep = string_to_value[s2]
                if s1_base == s2_base:
                    return

                for child in base_to_strings[s2_base]:
                    child_base, child_base_rep = string_to_value[child]
                    string_to_value[child] = (s1_base, child_base_rep * s1_base_rep / s2_base_rep / res)
                    base_to_strings[s1_base].add(child)
                base_to_strings[s2_base] = set()  # 해당 set은 이미 다 처리돼서 다른 key에 대한 set으로 넘어갔으니 clear 해준다.
            else:
                string_to_value[s2] = (s1_base, s1_base_rep / res)
                base_to_strings[s1_base].add(s2)

        for i in range(len(equations)):
            s1, s2 = equations[i]
            res = values[i]

            if s1 in string_to_value and s2 in string_to_value:
                if len(base_to_strings[s1]) > len(base_to_strings[s2]):
                    update(s1, s2, res)
                else:
                    update(s2, s1, 1/res)
            
            elif s1 in string_to_value:
                update(s1, s2, res)
            elif s2 in string_to_value:
                update(s2, s1, 1/res)
            else:
                string_to_value[s1] = (s1, 1)
                base_to_strings[s1].add(s1)
                update(s1, s2, res)
        
        outputs = []
        for q1, q2 in queries:
            if not(q1 in string_to_value and q2 in string_to_value):
                outputs.append(-1.0)
                continue
            q1_base, q1_rep = string_to_value[q1]
            q2_base, q2_rep = string_to_value[q2]
            if q1_base != q2_base:
                outputs.append(-1.0)
                continue
            outputs.append(q1_rep / q2_rep)
        
        return outputs
```



</details>


근데 이거를 그래프로 이해할 수도 있다. 예를 들어 a/b = 2 인 경우 a에서 b로 가는 edge가 있고 그 weight가 2라고 표현할 수도 있는 것이다.   
directed graph이고 화살표 방향이 나누는 방향이므로 중요하다.   
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
            visited.remove(curr_node)  # 되돌아가는 로직이다.
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

solution에 있는 union find 기법이다.   
좀 더 근본이다.   
union, find 라는 함수를 진짜로 사용을 했고 약간의 변형을 준 것이다.   
아이디어만 빌린 내 solution하고 다르다.   
좀 더 정형화된 규격에 맞춰진 솔루션 느낌이다.

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
                gid_weight[node_id] = \
                    (new_group_id, node_weight * group_weight)
            return gid_weight[node_id]

        def union(dividend, divisor, value):
            dividend_gid, dividend_weight = find(dividend)
            divisor_gid, divisor_weight = find(divisor)
            if dividend_gid != divisor_gid:
                # merge the two groups together,
                # by attaching the dividend group to the one of divisor
                gid_weight[dividend_gid] = \
                    (divisor_gid, divisor_weight * value / dividend_weight)

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







### 952. Largest Component Size by Common Factor

https://leetcode.com/problems/largest-component-size-by-common-factor/



