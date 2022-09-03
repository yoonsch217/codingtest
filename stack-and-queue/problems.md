### 739. Daily Temperatures

https://leetcode.com/problems/daily-temperatures

문제: temperatures라는 리스트가 있는데 하루 간격의 기온이 저장되어 있다. 각 날짜에서 더 따뜻한 날이 올 때까지 기다려야하는 일수를 저장한 리스트를 반환하라. 더 따뜻한 날이 이후에 없다면 0을 저장하면 된다.

decreasing monotonic stack을 사용한다.   
리스트를 iterate하면서 stack에는 index만 저장을 하게 되는데 현재 기온이 stack의 마지막보다 높다면 현재 기온보다 높은 날이 나올 때까지 pop을 한다.   
pop하는 원소는 그 index와 현재 index의 차이만큼 기다리면 현재 기온을 만나게 되는 것이므로 answer 리스트에는 그 index 차이를 저장한다.   
각 원소에 대해 한 번씩만 작업을 하게 되므로 O(N) 시간이 걸리게 되고 stack을 위한 O(N) 공간이 필요하게 된다.   

혹은 리스트를 뒤에서부터 iterate하면서 현재 날짜의 기온보다 높은 기온이 나오는 날을 찾는 방법도 있다.   
지금까지의 가장 높은 기온을 저장하는 hottest variable을 두고 현재 기온이 hottest보다 높다면 hottest를 업데이트하고 continue한다.   
이렇게 하는 이유는 그런 경우 더 따뜻한 날이 나올 수 없으므로 추가 작업이 필요 없기 때문이다.   
answer list를 만들어 놓고 뒤에서부터 원소를 하나씩 보는데, i번째 날에 i+1의 온도를 확인한다.   
i+1의 온도가 더 낮다면 i+1+answer[i+1] 위치로 가서 또 비교한다.   
더 높은 온도가 나올 때까지 반복을 하는데 이렇게 하면 각 원소마다 두 번씩만 작업을 하게 된다.(backward iterate할 때 한 번, jump하면서 날 찾을 때 한 번)   
따라서 O(N) time에 O(1) space 답을 해결할 수 있다.   



### 42. Trapping Rain Water

https://leetcode.com/problems/trapping-rain-water/



### 856. Score of Parentheses

https://leetcode.com/problems/score-of-parentheses/

문제: balanced parentheses string s가 주어졌을 때 점수를 구하라. () 형태로 붙어 있는 짝이 1점이고 (A) 처럼 감싸고 있으면 A*2이다. AB처럼 연속되어 있으면 A+B이다.

먼저 들었던 생각은 recursion하게 푸는 것이었다. 맨 처음은 left parenthesis일테니까 그 ( 에 대응하는 )로 한번 자른다. `helper(s) = 2*helper(s[1:k]) + helper([k+1:])`     
인덱스가 하나 차이난다면 1점을 return한다. 인덱스가 하나 넘게 차이난다면 안에 더 있는 거니까 helper(left+1, right-1) * 2 를 return한다.   
본 작업 전에 linear하게 훑으면서 각 괄호에 매칭하는 괄호 인덱스를 저장하면 O(N) 시간에 풀 수 있다.   
그런데 이 방법은 예외 처리가 조금 필요하다. 

stack을 사용해서 풀 수도 있다. 각 뎁스마다 값을 저장하는 것이다.   
left parenthesis 나올 때마다 depth가 늘어나니까 stack에 추가하고 right parenthesis 나올 때마다 depth 하나 탈출하니까 pop을 하면서 이전 depth의 값에 추가해준다.   

```python
def solve(s: str) -> int:
    s2 = [0]
    for i, c in enumerate(s):
        if c == '(':
            s2.append(0)
        else:
            tmp = s2.pop()
            if s[i-1] == '(':
                tmp += 1
            else:
                tmp = tmp*2
            s2[-1] += tmp

    return s2[-1]
```

마지막 방법은 power로 생각하는 것이다. 특정 depth에 있는 ()는 밖으로 나올 때마다 2가 곱해진다.   
그러면 왼쪽부터 linear하게 탐색하면서 열릴 때마다 depth를 증가시킨다. 닫힐 때 depth를 확인해서 pow(2, depth)를 결과에 더해준다. 바로 붙어있는 괄호들에 대해서만 처리하면 되는 듯.