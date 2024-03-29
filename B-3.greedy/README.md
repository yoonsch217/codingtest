

지역적으로 최적의 답이 모였을 때 전역적으로 최적의 답이 나올 때 사용한다.

아래의 두 가지 조건을 만족해야한다.

- 탐욕적 선택 속성(Greedy Choice Property) : 앞의 선택이 이후의 선택에 영향을 주지 않는다.
- 최적 부분 구조(Optimal Substructure) : 문제에 대한 최종 해결 방법은 부분 문제에 대한 최적 문제 해결 방법으로 구성된다.

아래 절차로 풀이를 접근한다. 최소의 동전 수로 amount를 만든다고 했을 때

- 선택 절차(Selection Procedure): 현재 상태에서의 최적의 해답을 선택한다. 비싼 동전일수록 적게 사용할 수 있으므로 가장 비싼 500원짜리를 고른다.
- 적절성 검사(Feasibility Check): 선택된 해가 문제의 조건을 만족하는지 검사한다. 500원을 선택했을 때 target amount를 넘지 않는지 확인한다. 넘으면 다시 1번으로 가서 동전을 고른다.
- 해답 검사(Solution Check): 원래의 문제가 해결되었는지 검사하고, 해결되지 않았다면 선택 절차로 돌아가 위의 과정을 반복한다. 동전을 골랐으면 그 값이 amount인지 확인한다.

그런데 이거는 특수한 케이스인 것 같다. 만약 동전이 20, 30만 있고 70을 만든다고 하면 30 + 30 을 골라버리게 되고 그 다음에는 답을 못 구하게 된다.   
이럴 땐 dp를 쓰는 게 맞을 것이다. dp의 coin change 문제 참고.


Graph의 Kruskal's algorithm, Prim's algorithm 등이 greedy를 사용한다.


