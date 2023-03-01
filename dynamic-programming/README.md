## 개념

recursive 알고리즘에서 겹치는 subproblem을 찾는 것이 중요하다. 겹치는 부분의 결과를 저장해서 다음 호출 때 사용한다.   
recursive 대신 iterative하게 구현하여 previous work를 캐싱하는 것도 가능하다.   


### 다른 알고리즘과 비교

divide and conquer과의 비교: 둘 다 subproblem을 풀긴 하지만 divide and conquer의 경우는 subproblem을 여러 번 풀지 않는다.   
greedy 의 경우도 optimal substructure를 갖지만 서로 겹치지 않는다.   


### dimension

state variable 이 n 개면 n-dimensional 이다.   
memoization을 사용할 때 memo 공간이 n-dimensional array가 된다.   
tabulation을 사용할 때 for loop이 n 개이다.


### tabulation & memoization

tabulation은 base case부터 시작을 한다.   
recursion이 갖는 오버헤드가 없어서 일반적으로 더 효율적이다.   
겹치는 연산이 없도록 설계해야한다.   

memoization은 array나 hashmap을 사용한다.   


### Dynamic Programming을 사용하는 경우

- Optimal value를 묻는 경우
maximum, minimum, how many ways, longest possible의 문제일 경우 => dynamic programming 혹은 greedy 문제일 가능성이 있다.
- future decision이 earlier decision에 영향을 받는 경우
greedy와의 차이점이다. house robber의 경우 house[i]를 선택하면 house[i+1]을 선택할 수 없으니 이후 결정에 영향을 준다.


### state reduction

state variable 의 수를 줄임으로써 효율적인 알고리즘을 만들 수 있다.   
일반적으로 bottom up의 경우는 time, space 둘 다 줄일 수 있고 top down의 경우는 time을 줄일 수 있다.   
state reduction에는 일반화될 수 있는 방법은 없고 잘 관찰해야한다.   
변수끼리 상관관계가 있다면 합칠 수 있는지 생각해본다.   
혹은 전체 상태를 저장하고 있었는데 실제로는 최근 n개의 상태만 필요하지 않은지도 확인해본다.   


### Counting dp

min, max를 사용하지 않고 count를 합치는 형태의 dp 문제이다.   
number of ways를 구하는 문제로 나올 수 있다.   
다른 dp들과 다르게 base case나 out of bound case가 보통은 0이 아니다.


### Kadane's Algorithm

integer array가 주어졌을 때 maximum sum subarray를 구하는 알고리즘이다.   
O(N) time, O(1) space만에 구할 수가 있다.   
array를 iterate하면서 각 index마다 이전까지의 결과를 갖고 갈지 버릴지를 결정한다.

1. best = negative infinity
2. current = 0
3. for num in nums:
    3.1. current = Max(current + num, num)
    3.2. best = Max(best, current)
4. return best

이게 모든 case를 다 cover할까? 
i ~ j 의 범위가 답이라고 해보자. 그럼 ? ~ i-1과 j+1 ~ ? 는 음수일 것이다. 
그렇다면 앞에서부터 iterate할 때 index가 i-1까지 가고 i에 도달하게 되면 지금까지의 답을 버리게 된다. 
따라서 i ~ j 구간을 cover하게 된다.


## 전략

- state variable을 찾고 dp(i)의 표현식을 찾는다. 여러 상황에 대한 합으로 만들어질 수 있으니 상황 분석을 잘 하자.   
- base case를 찾는다.   


### Iteration in the recurrence relation

dp 식에 for loop이 들어간다.    
min cost climbing stairs 문제에서 한 번에 두 칸이 아니라 k 칸이라면 k 개의 케이스를 고려한 뒤 최소를 택해야한다.

### do something vs do nothing

각 step마다 특정 상황에 따라 액션을 취할 때가 있고 아무것도 안 하고 다음 step으로 넘어갈 때도 있다.   
예를 들어 buy and sell stock 문제에서, stock을 구매한 상태라면 dosomething은 파는 거고 donothing은 다음 index로 넘어가는 것이다.   

### i의 방향 정하기

https://leetcode.com/problems/maximal-square 문제에서 dp(i,j)는 (i,j)에서 왼쪽 위를 바라본 최대 사각형이어야한다.   
2차원 매트릭스를 iterate할 때 일반적으로 왼쪽위부터 내려오는데 dp(i,j)는 이전의 결과를 이용해야한다.   
오른쪽아래를 바라보면 지금까지 구한 걸 이용하지 못하기 때문에 왼쪽 위를 바라보도록 해야한다.   
즉, 지금까지의 결과값을 사용해야하는 경우는 iterate 해온 방향을 바라보아야한다.


https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/ 문제는 최대 k transaction을 해야할 때 남길 수 있는 최대 이윤을 구하는 것이다.   
이전의 결과가 지금의 선택에 영향을 주긴 하지만 이전의 결과를 사용하지는 않는다.   
이전 작업으로 k 값이 바뀌는 것 뿐이다.   
이럴 때는 iterate 할 방향을 바라보도록 i를 잡는다.

