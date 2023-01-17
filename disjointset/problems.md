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


