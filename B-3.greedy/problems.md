

### 881. Boats to Save People

https://leetcode.com/problems/boats-to-save-people/

문제: people 리스트가 주어지는데 각 값은 몸무게이다. 보트들에 사람을 최대 두 명 태울 수 있는데 `limit` 의 무게를 넘을 순 없다. 모든 사람을 태우기 위한 최소의 보트 수를 구하라.
limit 보다 무거운 사람은 없다.

<details><summary>Approach 1</summary>

먼저 정렬을 한다. 그러고 two pointers로 오른쪽부터 하나, 왼쪽부터 하나를 놓고 이동한다.   
오른쪽부터 무거운 사람을 태우면서 left pointer 몸무게가 더 태울 수 있는 무게면 태우고 안 되면 만다.    
right pointer 이동할 때마다 보트를 사용하는 거니까 결과값을 1 늘린다.

나는 처음에 가장 무거운 사람의 짝으로, limit을 안 넘는 가장 무거운 사람을 골랐었다. 이렇게 하면 N^2 의 시간이 필요하다.   
하지만 지금 문제의 경우는 인원수 limit이 있기 때문에 매 순간 보트를 꽉꽉 채워서 보내지 않아도 된다. 이럴 때는 같이 탈 수 있는 사람을 누구라도 골라서 태워 보내면 되기 때문에 left pointer, right pointer 를 두고 선형적으로 탐색할 수 있다.    
무거운 것부터 A, B, C, D 사람이 있었을 때: A+C 와 A+D 모두 가능하다고 해보자. 그럴 때 A+C가 가능하다면 B+C도 가능하다. 따라서 A는 D랑 바로 짝을 지으면 된다.


```py
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        l, r = 0, len(people) - 1
        cnt = 0
        while l < r:
            if people[r] + people[l] <= limit:
                l += 1
            r -= 1
            cnt += 1
        if l == r:
            cnt += 1
        
        return cnt
```

이 방법은 at most 두 명까지 태울 수 있어서 가능한 것 같다. 두 명 제한이 없고 weight sum limit만 있다면? 그러면 처음 보낼 때부터 최대한 빡빡하게 무게를 채워서 보내야할 것이다. 


</details>





### 279. Perfect Squares

https://leetcode.com/problems/perfect-squares/

문제: 어떤 int n이 주어졌을 때 perfect square로만 합쳐서 n을 만들도록 할 때의 perfect square number의 최소의 개수를 구하라. perfect square는 정수의 제곱이다.

dp에 정리해놨다.

<details>

BFS 를 사용하는 방법이 가장 직관적인 것 같다.
n 보다 작은 perfect square 를 모두 구해놓고 각 iteration 마다 가능한 경우의 수를 구한다.
각 iteration 지날 때마다 count 가 하나 늘어난다.
n 이 만들어지는 순간의 count 가 답이다.


</details>








### 11. Container With Most Water

https://leetcode.com/problems/container-with-most-water/description/

문제: You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
Find two lines that together with the x-axis form a container, such that the container contains the most water.
Return the maximum amount of water a container can store.

<details><summary>Approach 1</summary>

Greedy한 접근을 생각해본다.   
너비와 높이가 중요하고, 하나를 포기하게 되면 다른 하나는 더 좋아져야한다.   
너비를 가장 넓게 시작을 해본다. 그러면 left와 right를 양 끝으로 잡는다.    
거기서 left와 right를 가운데로 움직이면 height는 무조건 높아져야한다.    
left와 right 중 낮은 wall을 갖는 걸 옮겨야한다. 높은 wall을 갖는 걸 옮겨봤자 height = min(left, right)이기 때문에 이 값이 높아질 수가 없다.

Greedy approach 증명
- Greedy Choice Property: elimination strategy + opportunity cost
  - 매 순간 낮은 쪽을 옮기는 게 나중에 후회할 일이 없다.
  - 높은 쪽을 옮겼을 때, min(left wall, right wall) 은 항상 left wall 보다 작거나 같다. left wall 은 고정이니까. 하지만 width 는 현재보다 좁아진다. 
  - 그러면 left wall 을 고정시킨 모든 케이스는 정답이 될 수 없기 때문에 left wall 을 고정시키는 경우를 버리는 것이다.
- Optimal Substructure
  - 전체 문제의 최적해는 left 를 옮기거나 right를 옮겨서 만든 부분 문제의 최적해 중 하나이다.


```py
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        ans = 0
        while left < right:
            l_height, r_height = height[left], height[right]
            cur = (right - left) * min(l_height, r_height)
            ans = max(cur, ans)
            if l_height > r_height:
                right -= 1
            else:
                left += 1
        
        return ans
```


</details>
