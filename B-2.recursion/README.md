## Backtracking

여러 솔루션들을 탐색하면서(Constraint Satisfaction Problems or CSPs) 후보 솔루션들을 만들어낸다. 
어떤 후보 솔루션이 답으로 갈 수 없음을 알게 되면 그 후보에서 파생되는 후보들은 모두 버린다.

subsequences/combinations/permutations 문제의 경우 backtracking을 고려해야한다.

개념적으로 DFS 비슷하다. 어떤 노드에서 자식 노드를 iterate하다가 어떤 노드가 valid solution으로 갈 수 없음을 알게 되면 그 노드 밑의 모든 노드를 버리고 부모 노드로 backtrack하여 다른 후보들을 본다.   
backtracking은 불필요한 탐색을 줄여주기 때문에 brute force보다 효율적이다.

예를 들어, 트리의 각 노드에 알파벳이 있고 root부터 내려오는 path가 단어를 이룬다고 하자. target word를 만드는 path를 찾고자 한다면, root에서 각 child path로 내려가보면서 비교한다. 
내려가다가 target word와 달라진다면 더이상 내려갈 필요 없이 다른 child path를 탐색하면 된다.



```python
def backtrack(candidate):
    if find_solution():
        output()
        return
    
    for next_candidate in list_of_candidates:
        if is_valid(next_candidate):  # 이 길이 정답이 될 가능성이 있는가?
            mark_decision()  # 이 조건에 대해 설정을 한다.
            backtrack()  # 더 탐색을 한다. solution에 한 step 더 가까이 간다.
            revert_decision()  # 추가한 설정을 제거해서 원 상태로 돌려놓는다.
```

예시: 52. N-Queens II


## Master Theorem

복잡도를 계산하기 위한 이론

다음과 같은 재귀식에만 적용이 가능하다.

- T(n) = aT(n/b) + f(n)
  - a: 분할된 문제의 개수 (a >= 1)
  - b: 문제를 나눌 때 크기가 줄어드는 비율 (b > 1)
  - f(n): 문제를 나누고 합치는 데 드는 비용

이 때, 재귀적으로 쪼개지는 노드의 총합은 n^(logb a) 이고, 현재 단계에서 밖에서 하는 일은 f(n) 이다. 따라서 이 두 개를 비교해야한다.   
- case 1: n^(logb a) > f(n)
  - T(n) = O(n^(logb a))
- case 2: n^(logb a) == f(n)
  - T(n) = O(n^(logb a) * log n)
- case 3: n^(logb a) < f(n)
  - T(n) = O(f(n)) 

