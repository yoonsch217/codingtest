## 개념

DP는 복잡한 문제를 더 subproblem 으로 나누고, 그 결과를 재사용하여 효율성을 높이는 알고리즘 설계 기법이다.

recursive 알고리즘에서 겹치는 subproblem을 찾는 것이 중요하다. 겹치는 부분의 결과를 저장해서 다음 호출 때 사용한다.   
recursive 대신 iterative하게 구현하여 previous work를 캐싱하는 것도 가능하다.

### Core Properties

- Overlapping Subproblems: 동일한 작은 문제들이 반복해서 계산되는 구조여야 한다. (예: 피보나치 수열)
- Optimal Substructure: 부분 문제의 최적 결과가 전체 문제의 최적 결과를 구성하는 데 사용될 수 있어야 한다.
  - 예시: 최단 경로 문제. 서울에서 부산까지의 최단 경로를 구하는 문제에서, 최단 경로가 대전을 거친다면 `(서울-부산 최단경로) = (서울-대전 최단경로) + (대전-부산 최단경로)` 가 된다. 


|                       | Divide and Conquer | Greedy    | Dynamic Programming |
|-----------------------| --                 | --        | -- |
| Overlapping Subproblems | 없음. 각 subproblem 이 독립적임.(disjoint)       | 없음. 여러 경로를 보지 않고 한 길만 보고 감.       | 필수. 동일한 계산이 반복되어야 dp 의 가치가 있음. |
| Optimal Substructure  | 보통 가짐. 쪼개진 문제들의 답을 합쳐서 해결함.         | 필수. 현재의 최선이 전체의 최선이어야 함.          | 필수. 부분해를 합쳐 전체 최적해를 구함. |
| 비고                    | 중복이 없기 때문에 기록을 할 필요가 없음.             | 미래를 보지 않고 직진함. 후진이 없음.             | 과거의 기록을 재사용 함. |



### Dynamic Programming을 사용하는 경우

- Optimal value를 묻는 경우
  - maximum, minimum, how many ways, longest possible의 문제일 경우 => dynamic programming 혹은 greedy 문제일 가능성이 있다.
- future decision이 earlier decision에 영향을 받는 경우
  - greedy와의 차이점이다. house robber의 경우 house[i]를 선택하면 house[i+1]을 선택할 수 없으니 이후 결정에 영향을 준다.




### State & Dimension

- State Variable: 문제를 정의할 때 변하는 값들이다. (knapsack 예: i번째 인덱스, 남은 무게 w)
- Dimension: 변수의 개수가 n개이면 n-차원 DP이다.
  - Space: 보통 O(N^k)의 배열이 필요하다. (k dimension)
  - Time: (상태의 개수) x (각 상태에서 수행하는 연산량) 으로 계산한다.
- memoization을 사용할 때 memo 공간이 n-dimensional array가 된다.   
- tabulation을 사용할 때 for loop이 n 개이다.




## 구현 방식 및 테크닉

DP 문제 접근
- 상태 정의 (State): dp[i]가 무엇을 의미하는지 문장으로 적어본다.
- 점화식 세우기 (Recurrence Relation): dp[i]를 만들기 위해 어떤 이전 상태들이 필요한지 정의한다.
- Base Case: 가장 작은 단위의 해를 정의한다. (예: dp[0], dp[1])
- 최적화: State Reduction이 가능한지 검토한다.


### Memoization & Tabulation

- Top-down (Memoization)
  - 방식: Recursive (재귀)
  - 특징: 필요한 subproblem만 계산한다. array 나 hashmap 을 활용한다. 
  - 장점: 가독성이 좋고, 문제 구조를 그대로 코드로 옮기기 편하다. 실제로 최종 답에 필요한 계산들만 한다. 
  - 단점: 재귀 깊이(Recursion Depth) 제한에 걸릴 수 있고, 함수 호출 오버헤드가 발생한다.
- Bottom-up (Tabulation)
  - 방식: Iterative (반복문)
  - 특징: base 부터 차례대로 표(Table)를 채워 나간다. 
  - 장점: recursion 이 갖는 오버헤드가 없어서 일반적으로 더 빠르고 메모리 효율적이다.
  - 단점: 반복문을 통해 배열의 0번 인덱스부터 끝까지 채우면서 모든 subproblem을 계산해야 하므로, 불필요한 계산이 포함될 수 있다.

이론적으로는 필요한 것만 계산한다는 점에서 Top-down이 효율적이어야 하지만, 실전에서는 Bottom-up이 더 선호되는 경우가 많다.
- 함수 호출 오버헤드: 재귀(Recursion)는 함수를 호출할 때마다 스택 메모리에 정보를 쌓는다. 이 과정이 단순 반복문(For-loop)보다 훨씬 무겁고 느리다.
- 재귀 깊이 제한 (Stack Overflow): 데이터가 커지면 재귀 호출이 너무 깊어져 프로그램이 터질 수 있다.
- 캐시 지역성 (Cache Locality): Bottom-up은 배열을 순차적으로 채우기 때문에 CPU 캐시를 아주 잘 활용한다. 컴퓨터 아키텍처 관점에서 훨씬 빠르다.
- 결론: 계산해야 할 상태가 전체의 극히 일부라면 Top-down이 유리하지만, 대부분의 상태를 확인해야 한다면 Bottom-up이 압도적으로 빠르다.




### State Reduction

state variable 의 수를 줄임으로써 효율적인 알고리즘을 만들 수 있다.   


- 핵심 아이디어: 현재 상태를 계산하기 위해 직전 k개의 상태만 필요하다면, 전체 배열을 유지할 필요가 없다.
- 방법
  - dp[i]를 구할 때 dp[i-1]만 쓴다면 변수 하나로 대체. 
  - dp[i][j]를 구할 때 dp[i-1] 행만 쓴다면 1차원 배열로 대체 (Sliding Window).
- 일반적으로 bottom up의 경우는 time, space 둘 다 줄일 수 있고 top down의 경우는 time을 줄일 수 있다.   
- state reduction에는 일반화될 수 있는 방법은 없고 잘 관찰해야한다.
  - 변수끼리 상관관계가 있다면 합칠 수 있는지 생각해본다.
  - 혹은 전체 상태를 저장하고 있었는데 실제로는 최근 n개의 상태만 필요하지 않은지도 확인해본다.


### Counting DP

- min, max 대신 sum을 사용하여 count를 합치는 형태의 dp 문제이다. 
- number of ways를 구하는 문제로 나올 수 있다.   
  - 예시: 1칸 또는 2칸씩 올라갈 수 있는 계단이 10개 있을 때, 올라가는 방법의 가짓수? 
  - 점화식: dp[i] = dp[i-1] + dp[i-2] 
- 다른 dp들과 다르게 base case나 out of bound case가 보통은 0이 아니다.


### Kadane's Algorithm (최대 부분합)

- integer array가 주어졌을 때 maximum sum subarray를 구하는 알고리즘이다.   
- O(N) time, O(1) space만에 구할 수가 있다.   
- array를 iterate하면서 각 index마다 이전까지의 결과를 갖고 갈지 버릴지를 결정한다.
- 점화식
  - f(i)를 `index i를 right end로 하는 subarray 중 가장 큰 sum 값` 이라고 하자.
  - `f(i) = max(nums[i], f(i-1) + nums[i])`
- 직관적 해석: 지금까지의 합에 나를 더하는 게 이득인가, 아니면 나부터 새로 시작하는 게 이득인가?


```py
best = negative infinity
buffer = 0  # 현재 포인터 i 기준에서 i-1을 right end로 하는 subarray 중 최대의 sum
for num in nums:
  buffer = max(buffer+num, num)  # 현재의 값인 num은 항상 포함이 되어야 하고 이전의 subarray를 사용할지 안 사용할지를 정해야한다. num이 포함되지 않은 케이스는 이전 iteration에서 이미 반영이 됐다.
  best = max(best, buffer)
```

이게 모든 case를 다 cover할까?

```
i ~ j 의 범위가 답이라고 해보자. 그럼 `? ~ i-1`과 `j+1 ~ ?` 는 음수일 것이다.
그렇다면 앞에서부터 iterate할 때 index가 i-1까지 가고 i에 도달하게 되면 지금까지의 답을 버리게 된다.
그러다가 index가 j를 넘어가는 순간 음수가 되므로 best는 업데이트 되지 못 한다.
따라서 i ~ j 구간을 cover하게 된다.
```

- Proof of Contradiction (귀류법)
  - i ~ j 의 범위가 답(최적해)이라고 할 때, i 이전의 합은 음수여야한다.
  - 결론 부정: 만약 i 바로 앞의 어떤 구간(예: i-1 에서 끝나는 구간)의 합이 양수라고 해보자.
  - 모순 발생: 그렇다면 [i, j] 구간에 그 양수 구간을 합친 것이 [i, j] 보다 더 큰 합을 가질 것이다. 이는 [i, j] 가 최적해라는 최초의 가정에 모순된다.
  - 결론: 따라서 i 이전의 누적합은 0보다 클 수 없다(음수 혹은 0이어야 한다).
- Mathematical Induction (귀납법)
  - Base Case: 첫 번째 원소에서 buffer는 첫 번째 값 그 자체이므로, 첫 번째 원소로 끝나는 최적해임이 자명하다.
  - 귀납 가정 (Inductive Step): k 번째 단계에서 buffer가 k 를 오른쪽 끝으로 하는 최적의 합을 들고 있다고 가정하자.
  - 단계 확장: k+1번째 원소를 만났을 때, 우리가 할 수 있는 선택은 두 가지뿐이다.
    - 이전의 최적(k까지의 합)에 나를 더한다. 
    - 이전 게 별로면(음수면) 나부터 새로 시작한다.
  - 이 중 큰 것을 택하므로, k+1 단계에서도 여전히 k+1을 끝으로 하는 최적의 합을 들고 있게 된다. 


https://leetcode.com/problems/maximum-sum-circular-subarray


### Knapsack algorithm

- 물건을 쪼갤 수 없을 때(넣거나 안 넣거나) 사용하는 패턴이다.
- dp[i][j]: 0번부터 i번째 물건까지 고려했을 때, 배낭의 용량이 j일 때 얻을 수 있는 최대 가치
- 공간 최적화: 1차원 배열로 풀 때는 뒤에서부터(reverse) 채워야 중복 선택을 방지할 수 있다.
- permutation 문제가 아니라 combination 문제인 경우는 다르게 접근해야한다. (eg. Coin Change 2)
  - 동전의 순서가 달라도 같은 case로 봐야하기 때문에 case를 셀 때 동전의 사용 순서를 강제해야한다.
  - 따라서 동전 종류에 대한 루프나 인덱스 j가 필요하다.
  - 순서를 강제하기 위해서 state 정의에 어디까지 고려했는가를 포함시켜야한다.
- 종류
  - 0/1 knapsack problem
    - 물건을 한 번만 사용한다. (partition equal subset sum)
    - 점화식: `dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i]] + value[i])`
    - 1차원 최적화 점화식: `dp[w] = max(dp[w], dp[w - weight[i]] + value[i]), 단 w는 역순으로 순회`
  - unbounded knapsack problem 
    - 물건을 무제한 사용한다. (coin change 2)
    - 점화식: `dp[i][w] = max(dp[i-1][w], dp[i][w - weight[i]] + value[i])`
      - 현재 행인 i 를 참조해야한다. 방금 담았던 무게를 또 담을 수 있다는 뜻이다.
    - 1차원 최적화 점화식: `dp[i][w] = max(dp[w], dp[w - weight[i]] + value[i]), 단 w는 정방향으로 순회`

todo: https://leetcode.com/problems/last-stone-weight-ii/description/
https://leetcode.com/discuss/study-guide/1152328/01-Knapsack-Problem-and-Dynamic-Programming





### Iteration in the recurrence relation

- 점화식 하나가 단 두 개의 이전 상태(i-1, i-2)만 보는 게 아니라, 가변적인 k개의 상태를 훑어야 할 때가 있다.
- 계단 오르기에서 한 번에 최대 k칸을 갈 수 있다면, 현재 위치 i에 도달할 수 있는 모든 출발점(i-1, i-2, ... , i-k)을 확인해야 한다.
- 점화식 내부에 for 루프가 들어가며, 시간 복잡도는 상태 개수(N)에 선택지 개수(k)를 곱한 O(N * k) 가 된다.

### do something vs do nothing

- 매 단계마다 적극적인 액션을 취할지, 아니면 현상 유지를 할지 결정하는 구조이다. 
- buy and sell stock 문제에서 stock 을 구매한 상태라면 오늘 내가 할 수 있는 선택은 '팔기' 혹은 '그대로 두기'이다. 
- `dp[i] = max(action result, dp[i-1])` 형태이다. 여기서 dp[i-1]은 어제까지의 최선의 상태를 그대로 이어받는(Do Nothing) 것을 의미한다.



### 여러 state를 갖는 문제 (State Machine DP)

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown

- 단순히 얼마나 벌었는지 수치 외에, 현재 내가 어떤 조건(State)에 놓여있는지가 중요할 때 사용한다.
- Best Time to Buy and Sell Stock with Cooldown 처럼 '구매 가능', '보유 중', '쿨다운 중'이라는 세 가지 상태가 서로 얽혀 있을 때이다.
- 각 상태를 독립적인 DP 배열(혹은 변수)로 관리한다.
  - hold[i]: i일에 주식을 들고 있을 때 최대 수익
  - sold[i]: i일에 주식을 팔았을 때 최대 수익
  - rest[i]: i일에 아무것도 안 하고 쉴 때 최대 수익
  - 이들은 서로의 전날 상태를 참조하며 업데이트된다.

### i의 방향 정하기


- DP의 핵심은 이미 계산된 값을 재사용하는 것이다. 따라서 i, j가 바라보는 방향은 내가 데이터를 훑어온 궤적과 일치해야 한다.
- Look-back
  - https://leetcode.com/problems/maximal-square : dp(i,j)는 (i,j)에서 왼쪽 위를 바라본 최대 사각형이어야한다.
  - Maximal Square 문제처럼 왼쪽 위에서 오른쪽 아래로 루프를 돈다면, dp[i][j]는 내 왼쪽(i, j-1), 위(i-1, j), 왼쪽 위(i-1, j-1)를 참조해야 한다. 그래야만 "지금까지 구한 사각형 중 가장 큰 것"을 확장할 수 있기 때문이다.
  - 즉, 지금까지의 결과값을 사용해야하는 경우는 iterate 해온 방향을 바라보아야한다.
- Transaction 방향
  - https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv : 최대 k transaction을 해야할 때 남길 수 있는 최대 이윤을 구하는 것이다.
  - Best Time to Buy and Sell Stock IV처럼 k 번의 거래 횟수가 중요한 경우, i 번째 날의 결정은 '남은 거래 횟수'라는 차원에 영향을 준다. 
  - 내가 지나온 과거의 거래가 현재의 가용 거래 횟수를 결정하므로, 진행 방향에 맞춰 상태를 정의해야 한다.
  - 이전의 결과가 지금의 선택에 영향을 주긴 하지만 이전의 결과를 사용하지는 않는다. 이전 작업으로 k 값이 바뀌는 것 뿐이다. 이럴 때는 iterate 할 방향을 바라보도록 i를 잡는다.


### Optimization on space(in grid problems)

grid 문제를 풀 때, grid 자체를 업데이트하면서 진행할 수 있으면 O(1)의 space complexity를 갖는다.




