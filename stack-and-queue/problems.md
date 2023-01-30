### 739. Daily Temperatures

https://leetcode.com/problems/daily-temperatures

문제: temperatures라는 리스트가 있는데 하루 간격의 기온이 저장되어 있다. 각 날짜에서 더 따뜻한 날이 올 때까지 기다려야하는 일수를 저장한 리스트를 반환하라. 더 따뜻한 날이 이후에 없다면 0을 저장하면 된다.

decreasing monotonic stack을 사용한다.   
리스트를 iterate하면서 stack에는 index만 저장을 하게 되는데 현재 기온이 stack의 마지막보다 높다면 현재 기온보다 높은 날이 나올 때까지 pop을 한다.   
pop하는 원소는 그 index와 현재 index의 차이만큼 기다리면 현재 기온을 만나게 되는 것이므로 answer 리스트에는 그 index 차이를 저장한다.   
각 원소에 대해 한 번씩만 작업을 하게 되므로 O(N) 시간이 걸리게 되고 stack을 위한 O(N) 공간이 필요하게 된다.   

<details>
    
```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        stack = []
        ans = [0] * n
        for i in range(n):
            if not stack:
                stack.append(i)
                continue
            while stack and temperatures[stack[-1]] < temperatures[i]:
                past_i = stack.pop()
                ans[past_i] = i - past_i
            stack.append(i)
        
        return ans
```
    
</details>

혹은 리스트를 뒤에서부터 iterate하면서 현재 날짜의 기온보다 높은 기온이 나오는 날을 찾는 방법도 있다.   
지금까지의 가장 높은 기온을 저장하는 hottest variable을 두고 현재 기온이 hottest보다 높다면 hottest를 업데이트하고 continue한다.   
이렇게 하는 이유는 그런 경우 더 따뜻한 날이 나올 수 없으므로 추가 작업이 필요 없기 때문이다.   
answer list를 만들어 놓고 뒤에서부터 원소를 하나씩 보는데, i번째 날에 i+1의 온도를 확인한다.   
i+1의 온도가 더 낮다면 i+1+answer[i+1] 위치로 가서 또 비교한다.   
더 높은 온도가 나올 때까지 반복을 하는데 이렇게 하면 각 원소마다 두 번씩만 작업을 하게 된다.(backward iterate할 때 한 번, jump하면서 날 찾을 때 한 번)   
따라서 O(N) time에 O(1) space 답을 해결할 수 있다.   

<details>

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        hottest = 0

        for i in range(n-1, -1, -1):
            cur = temperatures[i]
            if cur >= hottest:
                hottest = cur
                continue
            comp_idx = i + 1
            ans[i] += 1
            while cur >= temperatures[comp_idx]:
                ans[i] += ans[comp_idx]
                comp_idx += ans[comp_idx]
        
        return ans

```

</details>


### 42. Trapping Rain Water

https://leetcode.com/problems/trapping-rain-water/

문제: integer array가 주어지고 각 index의 값들은 그 index 위치에서의 bar 높이이다. 얼만큼의 물이 고일 수 있는지 구하라.

아이디어가 중요하다. brute force한 방법 먼저 생각해보자. 내가 처음 생각했던 건, 각 bar 기준으로 오른쪽으로 닿을 때까지 간 뒤에 닿게 되면 그만큼 물을 채우는 것이었다. 그런데 이렇게 하면 hole이 많아서 복잡해진다.   

현재 위치 i에서 물이 차려면 i 기준 왼쪽과 오른쪽 둘 다에 i보다 높은 bar가 있어야한다.    
`cur_trapped_water = min(left_max, right_max) - cur_height`
각 위치 i 기준으로 왼쪽에서 가장 높은 bar의 높이가 저장된 left_maxs와 오른쪽으로 한 결과인 right_maxs를 만든 뒤 답을 구한다.   
O(N) / O(N)

<details>

```python
    def trap(self, height: List[int]) -> int:        
        n = len(height)
        total = 0

        left_maxs = [0] * n  # i 기준 왼쪽 중에 가장 큰 값
        right_maxs = [0] * n
        left_max = right_max = 0
        for i in range(1, n):
            left_max = max(left_max, height[i-1])
            left_maxs[i] = left_max
        for i in range(n-2, -1, -1):
            right_max = max(right_max, height[i+1])
            right_maxs[i] = right_max

        for i in range(n):
            cur_trapped = min(left_maxs[i], right_maxs[i]) - height[i]
            if cur_trapped > 0:
                total += cur_trapped
        
        return total
```

</details>

위의 방법은 두 번 iterate해야하는데 stack을 쓰면 한 번의 iterate로 가능하다.    
decreasing monotonic stack을 만들면서 오른쪽으로 이동하는 것이다. 
그러다가 pop해야할 상황, 즉 현재 높이가 더 큰 상황이 발생하게 되고 pop 하고도 stack에 값이 남아있다면 pop하는 위치 기준으로 left bar와 right bar(current bar)가 존재한다는 의미이다.   
또한 left bar와 right bar 사이에 popped bar보다 높은 건 없고 popped bar 보다 낮은 영역은 이미 이전 작업에서 처리됐다. 따라서 left bar에서 right bar까지는 popped bar 높이로 평평하다고 가정할 수 있다. 

<details>

```python
    def trap(self, height: List[int]) -> int:        
        n = len(height)
        total = 0
        stack = []

        current = 0
        while current < n:
            while stack and height[current] > height[stack[-1]]:
                top = stack.pop()
                if not stack:
                    break
                distance = current - stack[-1] - 1
                bounded_height = min(height[current], height[stack[-1]]) - height[top]
                total += distance * bounded_height
            stack.append(current)
            current += 1
        
        return total
```

</details>


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


### 155. Min Stack

https://leetcode.com/problems/min-stack

문제: minStack이라는 클래스의 메소드를 구현하라. 일반 stack의 메소드들에 더해서 get_min 이라는 메소드를 갖는데 스택에 있는 최솟값을 반환한다. 모든 메소드는 O(1) 시간에 수행돼야한다.

스택은 계속해서 위로 쌓이는 자료구조이다. 어떤 최솟값이 있고 그 이후로 그보다 작은 값이 없다면 그 위의 모든 값들에 대해서는 get_min이 그 최솟값이다.   
따라서 stack에 (cur_val, min_so_far) 의 tuple을 넣어주면 된다. 

위 방법대로 하면 중복된 값이 많이 저장될 수 있다. 메모리 효율을 위해서 스택을 두 개 관리하는 방법도 있다. 하나는 그냥 스택, 다른 하나는 min 값이 바뀔 때만 저장하는 스택이다.   
따라서 pop을 할 때는 min stack의 위에 있는 값과 같으면 둘 다 pop을 하는 식으로 한다.   




