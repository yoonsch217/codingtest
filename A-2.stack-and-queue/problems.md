### 150. Evaluate Reverse Polish Notation

https://leetcode.com/problems/evaluate-reverse-polish-notation

문제: reverse polish notation으로 표현된 string인 tokens에 대해서 그 계산값을 반환하라. 
operator는 `+, -, *, /` 를 사용한다. 나누기는 소수를 버리고 정수만 남는다.    
ex) `tokens = ["4","13","5","/","+"]` => `(4 + (13 / 5)) = 6`


<details><summary>Approach 1</summary>

간단하다. 쭉 iterate하면서 연산자가 아니면 stack push하고 연산자면 최근 두 개 pop 한 뒤 계산하면 된다.

lambda를 쓰면 다르게 풀 수도 있다.   

  
```python
def evalRPN(self, tokens: List[str]) -> int:
        
    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "/": lambda a, b: int(a / b),
        "*": lambda a, b: a * b
    }
    
    stack = []
    for token in tokens:
        if token in operations:
            number_2 = stack.pop()
            number_1 = stack.pop()
            operation = operations[token]
            stack.append(operation(number_1, number_2))
        else:
            stack.append(int(token))
    return stack.pop()
```

</details>






### 739. Daily Temperatures

https://leetcode.com/problems/daily-temperatures

문제: temperatures라는 리스트가 있는데 하루 간격의 기온이 저장되어 있다. 각 날짜에서 더 따뜻한 날이 올 때까지 기다려야하는 일수를 저장한 리스트를 반환하라. 더 따뜻한 날이 이후에 없다면 0을 저장하면 된다.

<details><summary>Approach 1</summary>

decreasing monotonic stack을 사용한다.   
stack에는 아직 더 따뜻한 날을 못 만난 day가 저장되어 있다. 그러면 bottom에서 top으로 갈수록 덜 따뜻하다.   
리스트를 iterate하면서 지금 보는 기온이 top보다 낮으면 그냥 push한다.   
top보다 높으면 더 높은 top이 나올 때까지 pop하면서 pop된 날짜에 대해 답을 넣어준다.
답은 현재 보는 index와 pop된 날짜의 차이이다.

O(N) / O(N)

    
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

<details><summary>Approach 2</summary>

리스트를 뒤에서부터 iterate하면서 현재 날짜의 기온보다 높은 기온이 나오는 날을 찾는 방법도 있다.   
지금까지의 가장 높은 기온을 저장하는 hottest variable을 두고 현재 기온이 hottest보다 높다면 hottest를 업데이트하고 continue한다.   
이렇게 하는 이유는 그런 경우 더 따뜻한 날이 나올 수 없으므로 추가 작업이 필요 없기 때문이다.   
answer list를 만들어 놓고 뒤에서부터 원소를 하나씩 보는데, i번째 날에 i+1의 온도를 확인한다.   
i+1의 온도가 더 낮다면 i+1+answer[i+1] 위치로 가서 또 비교한다.   
더 높은 온도가 나올 때까지 반복을 하는데 이렇게 하면 각 원소마다 두 번씩만 작업을 하게 된다.(backward iterate할 때 한 번, jump하면서 날 찾을 때 한 번)   

O(N) / O(1)


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
            while cur >= temperatures[comp_idx]:
                comp_idx += ans[comp_idx]
            ans[i] = comp_idx - i
        
        return ans

```

</details>


### 42. Trapping Rain Water

https://leetcode.com/problems/trapping-rain-water/

문제: integer array가 주어지고 각 index의 값들은 그 index 위치에서의 bar 높이이다. 얼만큼의 물이 고일 수 있는지 구하라.


<details><summary>Approach 1</summary>

내 brute force한 방법   
- 앞에서부터 iterate하면서 left wall로 생각을 한다. 
- 각 left wall마다 오른쪽을 보면서 left wall 이상인 right wall을 찾는다. 그러면 물은 그 right wall을 넘지 못 하고 그 사이를 채운다.
- left wall 이상인 게 없다면 오른쪽 중 가장 높은 wall을 찾는다. 그러면 그 사이가 물이 채워진다.
- right wall을 찾으러 갈 때 각 right wall 후보와 left wall 사이에 얼만큼이 벽으로 채워져있는지 계산해놓는다.
- 그러면 마지막에 `width x min(left wall, right wall) - occupied` 를 하면 된다.
- 다음 iteration은 right wall이 left wall로 되는 상황부터 하면 된다.

이러면 O(N^2)의 시간이 소요돼서 느리다.

이걸 최적화하려면 `739. Daily Temperatures` 문제처럼 미리 각 left wall마다 그거보다 높은 wall이 처음으로 나오는 위치를 저장한 array, 
그 이후의 wall 중 가장 높은 높이를 저장한 array 두 개를 O(N) 시간에 만들어 놓으면 이후 작업도 O(N)에 가능할 것이다.    


```py
    def trap(self, height: List[int]) -> int:
        n = len(height)
        left = 0
        ans = 0
        while left < n-1:
            if height[left] == 0:  # If the height of left wall is 0, it cannot trap water.
                left += 1
                continue
            right = tallest_right = left + 1
            occupied, occupied_dict = 0, {}
            while right < n:
                occupied_dict[right] = occupied
                if height[right] >= height[left]:
                    break
                if height[right] > height[tallest_right]:
                    tallest_right = right
                occupied += height[right]
                right += 1
            if right == n:
                right = tallest_right
            width = right - left - 1
            ans += (width * min(height[left], height[right]) - occupied_dict[right])
            left = right
        return ans
```

</details>


<details><summary>Approach 2</summary>

아이디어를 생각하기 어려웠다.    
적분하듯이 쪼개서 각 위치에서의 물 양을 구한 뒤에 합하는 걸로 생각해보자.    
현재 위치 i에서 물이 차려면 i 기준 왼쪽과 오른쪽 둘 다에 i보다 높은 bar가 있어야한다.    
`cur_trapped_water = min(left_max, right_max) - cur_height`
각 위치 i 기준으로 왼쪽에서 가장 높은 bar의 높이가 저장된 left_maxs와 오른쪽으로 한 결과인 right_maxs를 만든 뒤 답을 구한다.   
O(N) / O(N)


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

<details><summary>Approach 3</summary>

위의 방법은 두 번 iterate해야하는데 decreasing monotonic stack을 쓰면 한 번의 iterate로 가능하다.    
- 오른쪽으로 iterate하면서 decreasing monotonic stack을 만든다. 그러면 stack에는 left wall 후보들이 남게 된다.
- stack 만들다가 pop해야할 상황, 즉 현재 높이가 stack의 top보다 높다면 pop을 한다. 그 pop된 위치의 bar는 자기보다 높은 left wall과 right wall이 있는 것이다.
- stack이 비게 된다면 left wall이 없으므로 무시한다.
- stack에 값이 남아있다면 stack의 top 값이 left wall이 된다. right wall은 current bar이다.
- left bar와 right bar 사이에 popped bar보다 높은 건 없으므로 popped bar 높이 윗부분인 `min(left bar, right bar) - popped bar * width` 만큼 물이 찰 수 있다.
- popped bar 보다 낮은 영역은 이미 이전 작업에서 처리됐다.    

어렵다. 신박하다.


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

문제: balanced parentheses string s가 주어졌을 때 점수를 구하라. () 형태로 붙어 있는 짝이 1점이다.
A와 B를 각각 subpart의 점수라고 할 때 (A) 처럼 감싸고 있으면 A*2이고 AB처럼 연속되어 있으면 A+B이다.   
`"(())"` => 2점, `"()()"` => 2점


<details><summary>Approach 1</summary>

stack을 사용해서 풀 수 있다.   
string을 iterate하면서 괄호 혹은 계산된 숫자를 stack에 넣는다. 
그러면 제일 마지막에는 결괏값 하나만 stack에 존재하게 된다.

```py
    def scoreOfParentheses(self, s: str) -> int:
        stack = []
        for c in s:
            if c == '(':
                # left paranthesis면 stack에 추가만 한다.
                stack.append('(')
            if c == ')':
                # right paranthesis면 계산을 해야한다.
                left = stack.pop()
                if left == '(':
                    # stack의 제일 위에 open이 있었다면 현재의 close와 합쳐서 1을 넣는다.
                    stack.append(1)
                else:
                    # 숫자가 있었다면 그 숫자를 두 배한다. 
                    # stack에는 연속된 숫자가 없음이 보장되므로 그 다음의 pop은 open일 것이다. 합쳐서 2배해서 넣는다.
                    stack.pop()
                    stack.append(left * 2)
            # 각 iteration마다 stack의 top들에 연속된 숫자가 없도록 압축해준다. 
            if stack and stack[-1] != '(':
                tmp = stack.pop()
                if stack and stack[-1] != '(':
                    tmp += stack.pop()
                    stack.append(tmp)
                else:
                    stack.append(tmp)
        return stack[0]
```

</details>


<details><summary>Approach 2</summary>

stack의 다른 방법도 있다. 각 뎁스마다 값을 저장하는 것이다.   
left parenthesis 나올 때마다 depth가 늘어나니까 stack에 추가하고 right parenthesis 나올 때마다 depth 하나 탈출한다.    
depth 줄일 때마다 stack을 pop 한다. 
이전 depth의 값에 추가해준다.   


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

</details>


<details><summary>Approach 3</summary>

O(1) space

마지막 방법은 power로 생각하는 것이다. 특정 depth에 있는 ()는 밖으로 나올 때마다 2가 곱해진다.   
그러면 왼쪽부터 linear하게 탐색하면서 열릴 때마다 depth를 증가시킨다. 닫힐 때 depth를 확인해서 pow(2, depth)를 결과에 더해준다. 바로 붙어있는 괄호들에 대해서만 처리하면 되는 듯.

```py
    def scoreOfParentheses(self, s: str) -> int:
        depth = -1
        res = 0
        prev_left = -1
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
                prev_left = i
                continue
            if c == ')':
                if prev_left == i - 1:
                    res += 2 ** depth
                depth -= 1
        return res
```

</details>






### 155. Min Stack

https://leetcode.com/problems/min-stack

문제: minStack이라는 클래스의 메소드를 구현하라. 일반 stack의 메소드들에 더해서 get_min 이라는 메소드를 갖는데 스택에 있는 최솟값을 반환한다. 모든 메소드는 O(1) 시간에 수행돼야한다.   
MinStack의 member function: `push`, `pop`, `top`, `getMin`


<details><summary>Approach 1</summary>

스택은 계속해서 위로 쌓이는 자료구조이다. 어떤 최솟값이 있고 그 이후로 그보다 작은 값이 없다면 그 위의 모든 값들에 대해서는 get_min이 그 최솟값이다.   
따라서 stack에 (cur_val, prev_min) 의 tuple을 넣어주면 된다.   
push할 때와 pop할 때 self.min을 업데이트 해주면 된다.   
push할 때는 현재와 비교해서 더 작으면 min이 업데이트 되는 것이고, pop할 때는 pop값이 min하고 똑같으면 prev_min으로 업데이트 해야하는 것이다.

위 방법대로 하면 중복된 값이 많이 저장될 수 있다. 메모리 효율을 위해서 스택을 두 개 관리하는 방법도 있다. 하나는 그냥 스택, 다른 하나는 min 값이 바뀔 때만 저장하는 스택이다.   
따라서 pop을 할 때는 min stack의 위에 있는 값과 같으면 둘 다 pop을 하는 식으로 한다.   


```py
class MinStack:

    def __init__(self):
        self.min = math.inf
        self.stack = []  # list of (current value, min value before getting current value)


    def push(self, val: int) -> None:
        self.stack.append((val, self.min))  # min 업데이트 전에 넣어야 prev_min이 된다.
        self.min = min(self.min, val)
        

    def pop(self) -> None:
        val, prev_min = self.stack.pop()
        self.min = max(prev_min, self.min)  # pop 된 값이 있기 전의 최솟값이 prev_min이다. 이 값이 현재의 min보다 크다면 이 popped value가 push될 때 min이 업데이트 된 거니까 pop할 때도 업데이트가 된다.
        

    def top(self) -> int:
        return self.stack[-1][0]
        

    def getMin(self) -> int:
        return self.min
```

</details>









### 84. Largest Rectangle in Histogram

https://leetcode.com/problems/largest-rectangle-in-histogram/description/

문제: Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

<details><summary>Approach 1</summary>

내 solution: TLE    
- stack을 두고 (i, h) 값을 넣는다. 인덱스 i 이후부터 지금까지 가장 낮은 wall의 높이는 h인 것을 보장한다. 
- i가 늘어났는데 h가 작아진다면 의미가 없다. 따라서 stack은 monotonic stack으로서 h 값이 점점 커져야한다.
- 리스트를 traverse하면서 right end는 현재 index로 잡는다. 
- 현재 높이가 stack에 있는 높이보다 작다면 stack에서 현재 높이보다 큰 값들을 다 pop한다. right end 높이보다 큰 값들은 더 이상 쓰이지 못 하기 때문이다. 
- stack을 구성하면 그 stack을 iterate하면서 `ans = max(ans, (right_end - stack_index) x stack_height )` 로 계산한다. 현재 자리(right_end) 기준으로 stack_index까지 중 가장 높은 공통 높이는 stack_height이기 때문이다.


Time Complexity: O(N^2). 최악의 경우 increasing stack이 만들어질 수 있다.



```py
    def largestRectangleArea(self, heights: List[int]) -> int:
        """
        From index idx, it is guaranteed that height is the shortest.
        If idx becomes larger, heigh with shorter value is no need. Only looks for higher value.
        """
        stack = []  # (idx, height), 
        ans = 0
        for i, height in enumerate(heights):
            last_idx = i
            while stack and height <= stack[-1][1]:
                last_idx, _ = stack.pop()
            stack.append((last_idx, height))

            for _idx, _height in stack:
                ans = max(ans, (i + 1 - _idx) * _height)
        
        return ans
```

</details>

<details><summary>Approach 2</summary>

어떤 지점 i를 기준으로, 해당 bar를 높이로 갖는 최대 rectangle을 구해보자. 그러면 해당 bar에서 왼쪽으로 봤을 때 처음으로 낮은 bar가 나오는 곳이 left index가 되고 반대가 right index가 된다.   
이렇게 각 i를 대상으로 하게 되면 모든 rectangle을 구할 수 있다.   

left_barriers를 생성한다. O(N). 739. Daily Temperatures 문제 생각하면 된다.   
right_barriers 생성한 뒤 이를 이용해서 답을 구한다.


```py
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        """
        # left_barrieres
        For each index i, left_barrieres[i] is the index of the nearest wall 
        that appears shorter than heights[i] on the left side.
        If not exist, -1
        """
        left_barriers = [-1] * n
        right_barriers = [n] * n

        for i in range(n):
            cur_h = heights[i]
            cmp_idx = i - 1
            while cmp_idx != -1 and heights[cmp_idx] >= cur_h:
                cmp_idx = left_barriers[cmp_idx]
            left_barriers[i] = cmp_idx
        
        for i in range(n-1, -1, -1):
            cur_h = heights[i]
            cmp_idx = i + 1
            while cmp_idx != n and heights[cmp_idx] >= cur_h:
                cmp_idx = right_barriers[cmp_idx]
            right_barriers[i] = cmp_idx
        
        ans = 0
        for i in range(n):
            cur_h = heights[i]
            left, right = left_barriers[i], right_barriers[i]
            ans = max(ans, (right - left - 1) * cur_h)
        
        return ans
```

약간의 최적화
- shortest 라는 변수 넣어서 shortest보다 작거나 같다면 shortest 업데이트하고 바로 넘어가기(옆 index랑 비교할 필요 없이)
- 오른쪽 iterate loop를 합치기

</details>






### 503. Next Greater Element II

https://leetcode.com/problems/next-greater-element-ii/description/

문제:



### 907. Sum of Subarray Minimums

https://leetcode.com/problems/sum-of-subarray-minimums/description/

문제:


<details><summary>Approach 1</summary>

brute force하게 N^2 Time 알고리즘을 생각했다.   

```py
    def sumSubarrayMins(self, arr: List[int]) -> int:
        n = len(arr)
        res = 0
        for i in range(n):
            cur_min = arr[i]
            for j in range(i, n):
                cur_min = min(cur_min, arr[j])
                res += cur_min
        
        return res % (pow(10,9) + 7)
```


</details>




<details><summary>Approach 2</summary>

- heap에 (value, index) 를 저장한다.
- 왼쪽부터 오른쪽으로 iterate한다. cur_idx가 subarray의 rightend라고 생각하자. 
- 0부터 first_min_index 까지가 leftend일 때는 min 값이 first_min_value이다.
- first_min_index+1 부터 second_min_index까지가 leftend일 때는 min 값이 second_min_value이다.
- 이렇게 해서 cur_index가 되면 break하고 rightend를 하나 더 늘린다.

근데 heap을 매번 pop, push를 반복해야 하기 때문에 이것도 결국 worst는 N^2이다.
구현 스킵

</details>


<details><summary>Approach 3</summary>

`503. Next Greater Element II` 를 활용해야한다.    

- 어떤 index를 기준으로 했을 때, 왼쪽에서 자기보다 처음으로 작은 값이 나오는 위치를 l, 오른쪽에 처음 나오는 위치를 r이라 하자.
- `arr[l+1:r] 범위에서는 어떤 contiguos subarray도 min 값은 arr[index]가 된다.
- duplicate value에 대한 처리를 고려해야한다.



</details>

###

https://leetcode.com/problems/sliding-window-maximum/description/

문제: 






https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/description/
https://leetcode.com/problems/online-stock-span/description/
https://leetcode.com/problems/maximal-rectangle/description/
https://leetcode.com/problems/remove-duplicate-letters/description/
