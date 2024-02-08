

### 881. Boats to Save People

https://leetcode.com/problems/boats-to-save-people/

문제: people 리스트가 주어지는데 각 값은 몸무게이다. 보트들에 사람을 최대 두 명 태울 수 있는데 `limit` 의 무게를 넘을 순 없다. 모든 사람을 태우기 위한 최소의 보트 수를 구하라.
limit 보다 무거운 사람은 없다.


먼저 정렬을 한다. 그러고 two pointers로 오른쪽부터 하나, 왼쪽부터 하나를 놓고 이동한다.   
오른쪽부터 무거운 사람을 태우면서 left pointer 몸무게가 더 태울 수 있는 무게면 태우고 안 되면 만다.    
right pointer 이동할 때마다 보트를 사용하는 거니까 결과값을 1 늘린다.

<details>

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

</details>

이 방법은 at most 두 명까지 태울 수 있어서 가능한 것 같다. 두 명 제한이 없고 weight sum limit만 있다면? 




### 279. Perfect Squares

https://leetcode.com/problems/perfect-squares/

문제: 어떤 int n이 주어졌을 때 perfect square로만 합쳐서 n을 만들도록 할 때의 perfect square number의 최소의 개수를 구하라. perfect square는 정수의 제곱이다.

dp에 정리해놨다.









### 11. Container With Most Water

https://leetcode.com/problems/container-with-most-water/description/

문제: You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
Find two lines that together with the x-axis form a container, such that the container contains the most water.
Return the maximum amount of water a container can store.

Greedy한 접근을 생각해본다.   
너비와 높이가 중요하고, 하나를 포기하게 되면 다른 하나는 더 좋아져야한다.   
너비를 가장 넓게 시작을 해본다. 그러면 left와 right를 양 끝으로 잡는다.    
거기서 left와 right를 가운데로 움직이면 height는 무조건 높아져야한다.    
left와 right 중 낮은 wall을 갖는 걸 옮겨야한다. 높은 wall을 갖는 걸 옮겨봤자 height = min(left, right)이기 때문에 이 값이 높아질 수가 없다.   


<details>


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
