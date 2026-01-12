### 150. Evaluate Reverse Polish Notation

https://leetcode.com/problems/evaluate-reverse-polish-notation

문제: reverse polish notation으로 표현된 string인 tokens에 대해서 그 계산값을 반환하라.
operator는 `+, -, *, /` 를 사용한다. 나누기는 소수를 버리고 정수만 남는다.    
ex) `tokens = ["4","13","5","/","+"]` => `(4 + (13 / 5)) = 6`   
reverse polish notation: 연산기호를 만나면 바로 전 두 개의 숫자로 연산을 함, The division between two integers always truncates toward zero.


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
            stack.append(int(token))  # to truncate to zero for divide operation
    return stack.pop()
```

</details>






### 739. Daily Temperatures

https://leetcode.com/problems/daily-temperatures

문제: temperatures라는 리스트가 있는데 하루 간격의 기온이 저장되어 있다. 각 날짜에서 더 따뜻한 날이 올 때까지 기다려야하는 일수를 저장한 리스트를 반환하라. 더 따뜻한 날이 이후에 없다면 0을 저장하면 된다.

- Input: temperatures = [73,74,75,71,69,72,76,73]
- Output: [1,1,4,2,1,1,0,0]

<details><summary>Approach 1</summary>

직접 머릿속에서 푼다고 생각해보면, 앞에서부터 쭉 보면서 높은 온도가 나오면 그 이전에 있던 낮은 온도들의 답을 즉각 알 수 있고 처리 대상에서 지워버린다. 그 온도보다 높았던 날들만 남겨놓고 이후를 탐색한다.     
이를 동일하게 구현한 게 decreasing monotonic stack 이다.   
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
            if cur >= hottest:  # 여기에 등호가 포함되어야 한다. 안 그러면 밑에 while 문에서 동일한 온도에서 무한루프에 갖힐 수 있다.
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

- Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
- Output: 6

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

아이디어를 생각하기 어렵다.    
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

위의 방법은 two pointer로 두 번 iterate해야하는데 decreasing monotonic stack을 쓰면 한 번의 iterate로 가능하다.   
이 스택 방식은 "웅덩이를 층별로 가로로 썰어서" 계산한다. 물이 고일 수 있을 때마다 그만큼을 미리 계산하고 버리는 거다.

알고리즘의 직관적 원리
- 감소할 때 (Descending): 벽의 높이가 낮아지는 동안은 물이 고일 수 없다. 나중에 '왼쪽 벽'이 될 후보들이므로 인덱스를 스택에 계속 쌓는다.
- 상승할 때 (Ascending): 현재 벽(current)이 스택의 top보다 높다면, 웅덩이의 바닥을 찾은 것이다.
- top = pop(): 방금 꺼낸 이 위치가 웅덩이의 바닥이 된다.
- 왼쪽 벽 찾기: 스택에 남은 그 다음 top이 왼쪽 벽이 된다. decreasing monotonic stack 에 따라서, 지금 top 이 가장 첫 번째의 왼쪽 높은 벽이다.
- 오른쪽 벽 찾기: 현재 조사 중인 current가 현재 벽을 pop 하게 만들었기 때문에 오른쪽 벽이 된다.

물의 양 계산 방식 (가로 층 단위 계산)
- 웅덩이는 바닥, 왼쪽 벽, 오른쪽 벽이 모두 있어야 형성된다.
- 가로 길이(Distance): 오른쪽 벽 인덱스 - 왼쪽 벽 인덱스 - 1
- 세로 높이(Bounded Height): min(왼쪽 벽 높이, 오른쪽 벽 높이) - 바닥 높이

어렵다. 신박하다.


```python
    def trap(self, height: List[int]) -> int:        
        n = len(height)
        total = 0
        stack = []
        current = 0
        
        while current < len(height):
            # 현재 벽이 이전 벽보다 높으면 웅덩이 처리 시작
            while stack and height[current] > height[stack[-1]]:
                top = stack.pop() # 웅덩이의 '바닥' 위치
                
                if not stack: # 왼쪽 벽이 없으면 물이 고일 수 없음
                    break
                
                # 웅덩이의 구성 요소 정의
                left_wall_idx = stack[-1]
                right_wall_idx = current
                
                # 1. 가로 길이: 양쪽 벽 사이의 거리
                distance = right_wall_idx - left_wall_idx - 1
                
                # 2. 세로 높이: 양쪽 벽 중 낮은 쪽까지만 물이 참 (이미 계산된 바닥 높이는 제외)
                bounded_height = min(height[left_wall_idx], height[right_wall_idx]) - height[top]
                
                # 3. 면적 추가
                total += distance * bounded_height
                
            stack.append(current) # 현재 벽을 다음의 왼쪽 벽 후보로 추가
            current += 1
        
        return total
```

</details>






### 856. Score of Parentheses

https://leetcode.com/problems/score-of-parentheses/

문제: balanced parentheses string s가 주어졌을 때 점수를 구하라. () 형태로 붙어 있는 짝이 1점이다.
A와 B를 각각 subpart의 점수라고 할 때 (A) 처럼 감싸고 있으면 A*2이고 AB처럼 연속되어 있으면 A+B이다.
- "()" has score 1.
- AB has score A + B, where A and B are balanced parentheses strings.
- (A) has score 2 * A, where A is a balanced parentheses string.
- `"(())"` => 2점, `"()()"` => 2점


<details><summary>Approach 1</summary>

stack을 사용해서 풀 수 있다. left 괄호는 그냥 push 하고 right 괄호가 나올 때만 처리한다.

```
직접 머리로 푼다고 생각하고 그걸 로직화시키는 연습
((()())())
((11)())  # for right prths, when the top is left prths, pop and push 1
((2)())  # for right prths, when the top is not left, pop and sum until left prths appears, and double the sum and push
(41)
```


```py
def scoreOfParentheses(self, s: str) -> int:
    stack = []
    for cur in s:
        if cur == '(':
            stack.append(cur)
        else:
            # 괄호가 서로 마주보고 있는 경우는 항상 1이 된다. 
            if stack[-1] == '(':
                stack.pop()
                stack.append(1)
            else:
                tmp = 0
                while stack and stack[-1] != '(':
                    tmp += stack.pop()
                stack.pop()
                stack.append(tmp * 2)
    res = 0
    while stack:
        res += stack.pop()
    return res

```

</details>


<details><summary>Approach 2</summary>

stack의 다른 방법도 있다. 각 뎁스마다 값을 저장하는 것이다.   
left parenthesis 나올 때마다 depth가 늘어나니까 stack에 추가하고 right parenthesis 나올 때마다 depth 하나 탈출한다.    
depth 줄일 때마다 stack을 pop 한다. 
이전 depth의 값에 추가해준다.   
이건 직관적이지가 않네. 좋은 방법인가?


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
그러면 왼쪽부터 linear하게 탐색하면서 열릴 때마다 depth를 증가시킨다. 닫힐 때 depth를 확인해서 pow(2, depth)를 결과에 더해준다.    
이건 수학이네. 각 항마다 독립적으로 계산하고 마지막에 더하기. 분배법칙?

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

stack이 있고, 특정 지점에서의 최솟값들을 저장한 min_stack이 있다고 하면 min_stack은 내림차순일 것이다.   
stack 을 탐색하다가 기존의 min 보다 작으면 그 값으로 채워지기 시작한다. 

이렇게 스택 두 개를 같은 길이로 사용하는 것보다 조금 더 메모리 효율적인 방법은 min_stack 에서 값이 바뀔 때만 append 하는 것이다. 그러면 pop 할 때, 두 stack이 동일할 때만 min_stack 에서 pop을 한다.


```py
class MinStack:
    def __init__(self):
        self._stack = []
        self._min_stack = []

    def push(self, val: int) -> None:
        self._stack.append(val)
        if not self._min_stack or val < self._min_stack[-1]:
            self._min_stack.append(val)
        else:
            self._min_stack.append(self._min_stack[-1])
        

    def pop(self) -> None:
        self._stack.pop()
        self._min_stack.pop()
        

    def top(self) -> int:
        return self._stack[-1]
        

    def getMin(self) -> int:
        return self._min_stack[-1]
        
```

</details>




### 84. Largest Rectangle in Histogram

https://leetcode.com/problems/largest-rectangle-in-histogram/description/

문제: Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.


<details><summary>Approach 1</summary>

접근을 잘 해야한다. 머릿속에서 풀더라도 어떻게 해야 최소한으로 건들고 풀 수 있을지를 생각하자.   
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

increasing monotonic stack 의 성질을 살린 solution

```py
def largestRectangleArea(self, heights: List[int]) -> int:
    left_ends = [-1] * len(heights)  # first index that has a lower height than the current index's height
    right_ends = [len(heights)] * len(heights)
    #  2,  1, 5, 6, 2, 3
    # -1, -1, 1, 2, 1, 4
    # 만약 increasing monotonic stack 이라면, 점점 증가하다가, 낮은 게 들어오면 다 pop
    # pop 되는 것 입장에서는 현재 값이 나보다 작은 첫 오른쪽 원소이다.
    # push 되는 것 입장에서는 top 값이 나보다 작은 첫 왼쪽 원소이다.

    stack = []
    for i, h in enumerate(heights):
        while stack and stack[-1][1] > h:
            prev_i, prev_h = stack.pop()
            right_ends[prev_i] = i
        if stack:
            left_ends[i] = stack[-1][0]
        stack.append((i, h))
    
    largest = 0
    for i, h in enumerate(heights):
        cur = ((right_ends[i]-1) - (left_ends[i]+1) + 1) * h
        largest = max(largest, cur)
    return largest



```

약간의 최적화
- shortest 라는 변수 넣어서 shortest보다 작거나 같다면 shortest 업데이트하고 바로 넘어가기(옆 index랑 비교할 필요 없이)
- 오른쪽 iterate loop를 합치기

</details>






### 503. Next Greater Element II

https://leetcode.com/problems/next-greater-element-ii/description/

문제: Given a circular integer array nums (i.e., the next element of nums[nums.length - 1] is nums[0]), return the next greater number for every element in nums. 
The next greater number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. 
If it doesn't exist, return -1 for this number.


<details><summary>Approach 1</summary>

monotonic stack을 활용한다.

- 뒤에서부터 iterate하면서 decreasing monotonic stack을 만든다.
- 현재 값보다 큰 값이 나올 때까지 pop을 한다. stack이 비게 되면 -1을 넣고 그렇지 않다면 top을 넣는다.
- 현재 값을 stack에 넣는다.
- circular list이기 때문에 이 과정을 두 번 반복한다. stack이 비게 되는 순간은 바로 break할 수 있다.

```py
def nextGreaterElements(self, nums: List[int]) -> List[int]:
    stack = []
    res = deepcopy(nums)
    for i in range(len(nums)-1, -1, -1):
        num = nums[i]
        while stack and num >= stack[-1]:
            stack.pop()
        if stack:
            res[i] = stack[-1]
        else:
            res[i] = -1
        stack.append(num)
    
    for i in range(len(nums)-1, -1, -1):
        num = nums[i]
        while stack and num >= stack[-1]:
            stack.pop()
        if stack:
            res[i] = stack[-1]
        else:
            break
        stack.append(num)
    
    return res
```

나는 그냥 두 배로 늘린 다음에 monotonic stack 한 번 쓰는 게 편했다.

```python
def nextGreaterElements(self, nums: List[int]) -> List[int]:
    linked_nums = nums + nums
    res = [-1] * len(nums)
    stack = []
    for i, num in enumerate(linked_nums):
        while stack and stack[-1][1] < num:
            prev_i, prev_num = stack.pop()
            if prev_i < len(nums):
                res[prev_i] = num
        stack.append((i, num))
    return res
```

</details>






### 907. Sum of Subarray Minimums

https://leetcode.com/problems/sum-of-subarray-minimums/description/

문제: Given an array of integers arr, find the sum of min(b), where b ranges over every (contiguous) subarray of arr. Since the answer may be large, return the answer modulo 10^9 + 7.    
arr = [3,1,2,4], Subarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4] => 17


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
- `arr[l+1:r] 범위에서는 어떤 contiguous subarray도 min 값은 arr[index]가 된다.
  - 결과를 iterate 하면서 각 index 위치마다 (left length * right length * nums[index]) 를 하면 된다.
  - 그 구간에서 만들어질 수 있는 모든 subarray는 index를 중심으로 왼쪽으로 left length 만큼 경우의 수가 있고 오른쪽으로 right length 만큼 경우의 수가 있기 때문이다.
- duplicate value에 대한 처리를 고려해야한다. 이게 어렵다.
  - 양 쪽 다 if greater, pop 을 하면 duplicate


```
both strictly greater condition

values: 3 1 2 3 2 4
index:  0 1 2 3 4 5

left:   N N 1 2 2 4
right:  1 N 4 4 N N

[2, 3, 2]로 extend 되는 건 빠진다.
```

```
both greater or equal condition

values: 3 1 2 3 2 4
index:  0 1 2 3 4 5

left:   N N 1 2 1 4
right:  1 N N 4 N N

[2, 3, 2]를 기준으로 extend 되는 건 2번씩 중복되게 된다.
```

```
left strictly greater, and right greater or equal condition

values: 3 1 2 3 2 4
index:  0 1 2 3 4 5

left:   N N 1 2 2 4
right:  1 N N 4 N N

[2, 3, 2]를 기준으로 extend 되는 건 왼쪽에 있는 게 담당한다.
오른쪽에 있는 값은 left를 빼고 extend하니까 중복이 안 된다.
```


어렵다.


```py
    def sumSubarrayMins(self, arr: List[int]) -> int:
        # number of moves from current position to reach the first less element
        left_dists = [0] * len(arr)
        right_dists = [0] * len(arr)

        d_stack = []
        for i, num in enumerate(arr):
            while d_stack and d_stack[-1][0] > num:
                d_stack.pop()
            if d_stack:
                left_dists[i] = i - d_stack[-1][1]
            else:
                left_dists[i] = i+1
            d_stack.append((num, i))

        d_stack = []
        for i in range(len(arr)-1, -1, -1):
            num = arr[i]
            while d_stack and d_stack[-1][0] >= num:
                d_stack.pop()
            if d_stack:
                right_dists[i] = d_stack[-1][1] - i
            else:
                right_dists[i] = len(arr) - i
            d_stack.append((num, i))
        
        res = 0
        #print(f"{len(left_dists)} {len(right_dists)}")
        for i in range(len(arr)):
            res += (left_dists[i] * right_dists[i] * arr[i])
        
        return res % (pow(10, 9) + 7)
```

</details>


<details><summary>Approach 4</summary>

Approach 2를 최적화한 방법이다. 
Approach 2의 경우는 각 원소당 최대 2 + 2 + 1 번 접근할 수 있을 것 같다.   
Approach 3는 각 원소 당 최대 2번 접근한다.

- 각 position i에 대해서, i가 rightend인 subarray를 생각하자.
- i가 i-1에서 i로 하나 증가하게 되면, i-1일 때의 subarray들에서 i만 append한 것에 `[i]` 만 추가된 것이다.
- arr[i]가 arr[i-1]보다 크다면, arr[i]가 추가된 것은 min에 영향을 주지 못한다.
- arr[i]가 더 작다면, 처음으로 arr[i]보다 작은 값이 나오는 곳을 찾는다. 그 구간까지는 min 값이 바뀌어야한다.
- result[i]를 i가 rightend인 subarray들의 min 합이라고 하자.
- if arr[i] >= arr[i-1], result[i] = result[i-1] + arr[i]
- else, result[i] = result[j] + arr[i] * (i-j), for j the first less element

코드도 훨씬 간단하다.


```py
    def sumSubarrayMins(self, arr: List[int]) -> int:
        arr = [0] + arr
        result = [0]*len(arr)
        stack = [0]
        for i in range(len(arr)):
            while arr[stack[-1]] > arr[i]:
                stack.pop() 
            j = stack[-1]
            result[i] = result[j] + (i-j)*arr[i]
            stack.append(i)
        return sum(result) % (10**9+7)
```

</details>









### 239. Sliding Window Maximum

https://leetcode.com/problems/sliding-window-maximum/description/

문제: You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.   
Return the max sliding window.


<details><summary>Approach 1</summary>

monotonic queue를 활용한다.   

- 큐에는 작아지는 순서로 데이터가 들어가게 된다.
- 새로운 값이 들어올 때, head를 하나씩 보면서 index가 유효하지 않으면 버린다.
- tail부터 지금의 값보다 작으면 버리고 지금 값의 위치를 찾아 들어간다.
- 버려진 값들은 지금 값보다 작으면서 왼쪽에 존재하는 거니까 앞으로의 답에 영향을 줄 수 없다.

```py
def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    m_queue = deque()
    res = []
    for i, num in enumerate(nums):
        while m_queue and m_queue[0][1] <= i - k:
            # 여기는 index만 봐도 된다. 남은 건 유효한 max니까.
            m_queue.popleft()
            
        while m_queue and (m_queue[-1][0] <= num or m_queue[-1][1] <= i - k):
            # 여기는 값만 비교해도 되긴 되는데 그냥 정리해주자.
            m_queue.pop()

        m_queue.append((num, i))
        if i < k-1:
            continue

        res.append(m_queue[0][0])
    
    return res
```

O(N) / O(N)

</details>





---




https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/description/
https://leetcode.com/problems/online-stock-span/description/
https://leetcode.com/problems/maximal-rectangle/description/
https://leetcode.com/problems/remove-duplicate-letters/description/
