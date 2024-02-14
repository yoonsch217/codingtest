### 875. Koko Eating Bananas

https://leetcode.com/problems/koko-eating-bananas

문제: piles 라는 리스트가 있고 각 원소는 바나나의 개수이다. 감시자가 h 시간동안 떠나있을 때, 시간당 k의 속도로 바나나를 먹는다. 한번에 하나의 pile만 먹을 수 있다. 전체 바나나를 다 먹기 위한 최소의 k를 구하라.

<details>

각 pile 먹는 속도는 math.ceil(pile/k) 이다. 따라서 총 걸리는 시간은 `sum of math.ceil(pile/k) for each pile in piles` 이 된다.    
전체 바나나를 다 먹을 수 있는 속도라면 possible, 못 먹으면 impossible이라고 하자.   
최대의 impossible k 보다 하나 크면 최소의 possible k가 되고, 그게 답이 된다.   
따라서 binary search로 풀 수 있다.    
left와 right를 정해야하는데 최소는 1이 되고 최대는 max(piles)가 된다.   

x x x o o o 형태이고 left condition(왼쪽 파트, 즉 x로 인식되는 조건))은 `k가 작아서 총 걸리는 시간이 h보다 클 때` 이다.   
while loop을 나왔을 때 답은 left가 된다.   


```py
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def get_total_hour(piles, k):
            total_hour = 0
            for pile in piles:
                total_hour += ceil(pile / k)
            return total_hour
        
        l, r = 1, max(piles)
        while l <= r:
            m = (l+r) // 2
            if get_total_hour(piles, m) > h:
                l = m + 1
            else:
                r = m - 1
        return l
```

</details>



### 1642. Furthest Building You Can Reach

https://leetcode.com/problems/furthest-building-you-can-reach

문제: heights, bricks, ladders 가 주어진다. 건물들을 왼쪽부터 이동하는데 높은 건물로 갈 때는 높이 차이만큼 brick을 쓰든가 ladder 하나를 써야한다. 가장 멀리 갈 수 있는 건물을 찾아라.


<details>

**Heap**    
ladders 개수 L 을 크기로 갖는 힙을 생성하고 앞에서부터 높이 차를 넣으면서 힙을 채운다. 즉, 사다리로만 올라가되 그 높이차를 기록해놓는 것이다.   
사다리를 다 썼을 땐 이제 벽돌을 써야한다. 벽돌 써야하는 상황 왔을 때, 현재 필요한 벽돌과 힙에 있는 최소의 높이차를 비교한다.     
현재 필요한 벽돌 수가 더 적으면 벽돌 소진하면 되는 것이고, 힙에 있는 최솟값이 더 작으면 예전에 썼던 사다리를 지금 쓰고 예전 작업은 벽돌로 하면 된다.   
이렇게 해서 벽돌 수가 음수가 되면 더이상 못 가는 것이다.   



```py
class Solution:
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        mheap = []
        prev_h = heights[0]
        pos = 0
        for i, h in enumerate(heights):
            diff = h - prev_h
            prev_h = h 

            if diff <= 0:
                continue

            if len(mheap) < ladders:
                heapq.heappush(mheap, diff)
                continue

            if mheap and diff > mheap[0]:
                heapq.heappush(mheap, diff)
                diff = mheap[0]
                heapq.heappop(mheap)
            bricks -= diff
            if bricks < 0:
                return i - 1

        return len(heights) - 1


```
    


**Binary Search**    

특정 위치까지 갈 수 있나 없나는 판단할 수 있다. 그 구간의 height diffs를 받아서 정렬한 뒤 min부터 벽돌 사용하도록 하면 판단이 된다.    
`0~threshold` 까지는 reachable 이고 `threshold+1 ~ end` 는 unreachable이다.    
이 특성을 이용해서 binary search를 사용할 수 있다.    
o o o x x x    => Get right pointer    


```py
    def furthestBuilding(self, heights: List[int], bricks: int, ladders: int) -> int:
        n = len(heights)
        l, r = 0, n-1

        def is_reachable(idx, bricks, ladders):
            h_diffs = []
            prev = 0
            for cur in range(1, idx+1):
                if heights[cur] > heights[prev]:
                    h_diffs.append(heights[cur] - heights[prev])
                prev = cur
            h_diffs.sort()
            for h_diff in h_diffs:
                bricks -= h_diff
                if bricks < 0:
                    ladders -= 1
                if ladders < 0:
                    return False
            return True

        while l <= r:
            m = (l + r) // 2
            if is_reachable(m, bricks, ladders):
                l = m + 1
            else:
                r = m - 1
        
        return r
```

매번 정렬을 하면 너무 cost가 크기 때문에 한번 정렬해서 height diff마다 position을 붙여놓는다. 그러고는 linear하게 iterate하면서 현재 찍은 기준 position보다 낮은 index를 가진 경우만 벽돌이나 사다리에서 차감한다. => 이건 다음에 구현해보자.   

O(N logN)


</details>







### 162. Find Peak Element

https://leetcode.com/problems/find-peak-element/

문제: 서로 다른 값을 갖는 integer array가 주어졌을 때 peak의 위치를 반환하라. peak란 주변보다 strictly greater한 값을 가지는 곳을 말하며 `nums[-1] = nums[n] = -math.inf` 으로 간주한다. O(log n) 의 알고리즘을 구하라.

<details>

left, right를 초기화하고 nums에 -math.inf 를 append한다. 이렇게 함으로써 `nums[-1] = nums[n] = -math.inf`를 자연스럽게 적용시킬 수 있다.   
mid 기준에서 왼쪽이 더 크면 left half에 peak가 있어야한다. nums[-1]은 -inf이고 nums[cur] 보다 nums[cur-1] 이 더 크기 때문이다. 오른쪽에도 있을 수 있지만 왼쪽엔 보장이 된다.   
오른쪽이 더 크면 right half에 peak가 있어야한다.   
둘 다 아니면 현재가 peak이다.


```py
    def findPeakElement(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 0
        nums.append(-math.inf)

        l, r = 0, len(nums) - 1
        while l <= r:
            m = (l + r) // 2
            
            if nums[m-1] > nums[m]:
                r = m - 1
            elif nums[m+1] > nums[m]:
                l = m + 1
            else:
                return m

"""
이건 딱 조건을 만족하는 포인트인 m을 찾는 문제이니까 o o o x x 이런 식으로 생각하지 않아도 된다. 
m은 ans 지점을 거치게 되고 if ans: return m 로 답을 구하면 된다.
"""
```

</details>




### 658. Find K Closest Elements

https://leetcode.com/problems/find-k-closest-elements/

문제: sorted integer array가 주어지고 k, x가 주어진다. x랑 가장 가까운 k개의 원소를 정렬된 순서로 반환하라. abs(y - x)가 동일하면 작은 y값이 더 가까운 걸로 간주한다. x가 arr에 존재하지 않을 수 있다.



<details><summary>brute force O(N)</summary>

```py
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        # 전체를 놓고 양 쪽을 비교하면서 not closer 한 곳을 줄인다.
        l, r = 0, len(arr) - 1
        while r - l > k - 1:
            if x - arr[l] <= arr[r] - x:  # 두 개가 같으면 right를 버려야한다. 값이 더 작은 게 우선이기 때문이다.
                r -= 1
            else:
                l += 1
        
        return arr[l: r+1]
```

</details>



<details><summary>내 첫 approach</summary>

너무 헤맸다.     
헤맨 포인트:
- x랑 가장 가까운 포인트를 찾아서 거기서부터 expand해야하는데 그 포인트를 찾는 게 어려웠다.
- [1, 4, 4, 4, 5] 와 같이 동일한 값이 연속이고 x가 2일 때, mid 값이 index 2에 있다면 어느 쪽을 search할지 정해야한다. index 2의 값이 4이고, x 값은 2이므로 더 작은 쪽을 봐야하므로 left를 search하도록 해아한다.
- mid-1 과 mid 중 mid-1이 더 가깝다면 mid+1은 생각할 필요 없이 left half를 탐색하면 된다.
- 아니라면 오른쪽도 비교한다. mid-1이나 mid+1 둘 다 mid보다 가깝지 않다면 mid가 가장 가까운 포인트이다.
- 이걸 is_closer 함수를 하나 만들어서 해야한다. 안 그러면 너무 코드가 복잡해진다.


기본 로직

- binary search를 사용하여 x에 가장 가까운 값을 찾는다.
  - is_first_param_closer 라는 함수를 하나 짜서 `is_first_param_closer(mid-1, mid)`, `is_first_param_closer(mid, mid+1)` 를 구해서 closer 쪽으로 search한다.
- 그 index를 기준으로 한 칸 왼쪽을 l, 한 칸 오른쪽을 r로 둔다.
- l과 r 중 더 가까운 쪽을 한 칸 더 옮긴다. 이 작업을 k번 반복한다.



```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        Find closest first.
        Expanding left and right, find k elements.
        """
        if k == len(arr):
            return arr
        
        def is_first_param_closer(l, r):
            if not 0 <= l < len(arr):
                return False
            if not 0 <= r < len(arr):
                return True
            l_val = arr[l]
            r_val = arr[r]
            if l_val == x:
                return True
            if l_val == r_val:
                if l_val < x:
                    return l > r
                return l < r
            if abs(l_val - x) == abs(r_val - x):
                return l_val < r_val
            return abs(l_val - x) < abs(r_val - x)
        
        left = 0
        right = len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            
            if is_first_param_closer(mid-1, mid):
                right = mid - 1
            elif is_first_param_closer(mid+1, mid):
                left = mid + 1
            else:
                break
        
        # Expand starting from x
        l, r = mid - 1, mid + 1
        for remained_iter in range(k-1, 0, -1):
            if l < 0:
                r += remained_iter  # (l, r) 이 답이다.
                break
            if r > len(arr) - 1:
                l -= remained_iter
                break
            
            if abs(x - arr[l]) == abs(x - arr[r]):
                if arr[l] < arr[r]:
                    l -= 1
                else:
                    r += 1
            
            elif abs(x - arr[l]) < abs(x - arr[r]):
                l -= 1
            else:
                r += 1
            
        return arr[l+1:r]
```

</details>




<details><summary>Best solution</summary>

- answer subarray의 시작 지점을 찾는 것이다.   
- 정답 array의 시작 지점으로 가능한 범위의 left bound는 0이고 right bound는 n-k이다. 
- 이 left, right에 대해 mid를 구한 후 mid 값과 mid + k 값을 비교한다. 
- mid ~ mid+k 의 subarray를 보면 크기가 k+1 이기 때문에 mid나 mid+k 중 하나는 버려져야한다. k+1로 하는 이유는 최적화된 subarray가 더 오른쪽에 있는지 왼쪽에 있는지 알아야하기 때문이다.      
- mid가 target에 더 가깝다면 `[mid+1, mid+k]` 은 답이 될 수 없고 그 이후의 subarray도 마찬가지이다. `[mid, mid+k-1]` 혹은 그 왼쪽의 subarray가 답이 될 수 있다.
- subarray의 시작지점은 mid 이하가 된다. right를 mid로 옮기고 다음 iteration을 진행한다.   
- 이걸 반복하다가 left == right 되는 순간이 답이다.




```py
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left, right = 0, len(arr) - k
        while left < right:
            mid = (left + right) // 2
            if x - arr[mid] > arr[mid + k] - x:  # head 값이 tail 값보다 멀다면 시작점은 0~mid 가 될 수 없다.
                left = mid + 1
            else:
                right = mid  # 아니라면 시작점은 0~mid 중 하나이다.
        return arr[left:left + k]

"""
condition: mid가 더 멀다 => mid+1 이후에 start point가 있다.
not condition: mid+k가 더 멀다 => mid 이전에 start point가 있다. mid가 답인지는 아직 모른다. 따라서 right를 mid로 바꿔주고 다음 작업을 해야한다.
"""
```


</details>





### 744. Find Smallest Letter Greater Than Target

https://leetcode.com/problems/find-smallest-letter-greater-than-target/

문제: 알파벳 소문자로 이루어진 리스트가 주어지고 target character가 주어진다. target char보다 큰 문자 중 가장 작은 문자를 반환하라. target char보다 큰 게 없으면 첫 번째 문자를 반환하라.

<details>

- o o o x x x
- left condition: target char보다 작거나 같다
- left 구한다. left가 index 밖이라면 첫 번째 문자를 반환한다.

```py
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        l, r = 0, len(letters) - 1
        while l <= r:
            m = (l + r) // 2
            if letters[m] <= target:
                l = m + 1
            else:
                r = m - 1 
        if l >= len(letters):
            return letters[0]
        return letters[l]
```

</details>







### 153. Find Minimum in Rotated Sorted Array

https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/

문제: sorted array with unique integer가 주어진다. 한 군데를 기준으로 rotate 되어 있는데 그 array의 최솟값을 구하라. `ex) [3, 4, 5, 1, 2] => return 1`


<details>

데이터를 먼저 이해하자. mid가 각 상황일 때 어떤 의미인지를 충분히 생각하고 깔끔하게 분류한 걸 글로 정리한 뒤에 코드를 생각해보자.

start < end면 rotated가 안 된거니까 그거에 대한 처리를 먼저 한다.    
rotated 된 경우에 대해서는 index 0의 value를 기준으로 잡는다. 아래와 같이 우선 데이터를 이해한다.
- `[index 1, rotated point-1]`: index 0의 값보다 큰 값이다.   
- `[rotated point, last]`: index 0의 값보다 작은 값이다.    

이해한 데이터를 바탕으로 아래 로직으로 찾는다. Using binary search
- nums[l] < nums[r] 라면 sorting 된 상태이기 때문에 nums[l] 반환
- mid가 nums[l]보다 작다면 mid 위치는 `[rotated point, r]` 범위의 값이다.   
  - mid보다 한 칸 낮은 nums[mid-1]이 nums[l]보다 크거나 같으면 mid-1은 `[rotated point, last]` 범위의 왼쪽이라는 뜻이다. 따라서 mid가 rotated point이다. => nums[mid]를 반환    
  - mid-1 이 out of index가 되려면 mid가 0이어야하는데 이런 상황은 발생하지 않는다. 이미 첫 번째 조건에서 반환되었을 것이기 때문이다.
  - 위 조건이 아니라면 mid는 `[rotated point, last]` 범위 중간에 있다는 것이고 mid 왼쪽에 rotated point가 있기 때문에 left half를 탐색한다. `right = m - 1`
- 위 조건이 아니라면 right half를 탐색한다. `left = m + 1`   


```py
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l <= r:
            if nums[l] <= nums[r]:
                return nums[l]
            
            m = (l + r) // 2
            if nums[m] < nums[l]:
                if nums[m-1] >= nums[l]:
                    return nums[m]
                r = m - 1
            else:
                l = m + 1
            
```

</details>







### 154. Find Minimum in Rotated Sorted Array II

https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/

문제: 중복된 값이 있는 integer array가 ascending order로 정렬되어 있었는데 몇 번의 rotation이 일어난 상태이다. 이 array에서 최솟값을 구하라.

<details><summary>내 풀이</summary> 

중복된 값이 있음으로서 달라지는 부분이 있다.   
로직은 다음과 같다.   
- 기본 binary search 템플릿을 사용한다.
- nums[left] < nums[right] 라면 그 subarray는 rotation이 없는 것이므로 nums[left] 를 반환한다.
- mid가 mid-1보다 작으면 mid가 rotation point이므로 nums[mid]를 반환한다. left가 최솟값이 아님을 보장하므로 mid-1은 항상 존재한다. 이 다음은 항상 rotated되었음을 가정하고 진행할 수 있다.
- mid가 left보다 작다면 rotation point가 왼쪽에 있을테니 left half를 탐색한다.
- mid가 left보다 크다면 rotation point가 오른쪽일테니 right half를 탐색한다. 
- mid가 left와 같을 때가 문제이다.
  - right가 left보다 작다면 right half를 탐색한다. right가 left보다 큰 건 이미 처리했다.
  - right가 left와 같을 때만 남았는데 이 때는 linear하게 탐색을 한다.


</details>



<details><summary>solution</summary> 

비슷한데 linear 탐색 대신 범위를 하나 좁혀서 다시 binary search를 시도한다.   
데이터 분석을 다시 해보자.

left를 기준으로 한 데이터 분석
- `[left, rotation point-1]`: left 보다 크거나 같아야한다.
- `[rotation point, rightend]`: left보다 작거나 같아야한다.

풀이
- if nums[left] < nums[right]:
   - return nums[left]
- if nums[mid] < nums[left]:
   - `[rotation point, rightend]` 범위에 mid가 있는 것이다. mid가 답인지 검증하고 아니라면 left half를 탐색해야한다.
- if nums[mid] > nums[left]:
   - right half를 탐색한다.
- if nums[mid] == nums[left]:
   - left를 하나 올려서 다시 탐색한다.



```py
def findMin(self, nums: List[int]) -> int:
    l, r = 0, len(nums) - 1

    while l <= r:
        if r - l <= 1:
            return min(nums[l], nums[r])
        if nums[l] < nums[r]:
            return nums[l]
        mid = (l + r) // 2

        if nums[mid] < nums[l]:
            if nums[mid-1] > nums[mid]:
                return nums[mid]
            r = mid - 1
        elif nums[mid] > nums[l]:
            l = mid + 1
        else:
            l += 1
```

</details>






### 240. Search a 2D Matrix II

https://leetcode.com/problems/search-a-2d-matrix-ii/

문제: 각 row와 column은 ascending order로 sort 되어 있다. target 이 해당 matrix 안에 있는지 판별하는 알고리즘을 구현하라.

<details>

- 어떤 위치에서 target보다 크다면, 해당 위치 기준 righter, lower elements는 다 무시할 수 있다. 이 때는 왼쪽으로 한 칸 이동해야한다.
- 어떤 위치에서 target보다 작다면, 해당 위치 기준 lefter, upper elements는 다 무시할 수 있다. 이 때는 아래로 한 칸 이동해야한다.
- binary search로 target을 찾으면 return True, 못 찾고 loop를 나오면 return False

```python
def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    m, n = len(matrix), len(matrix[0])
    row, col = 0, n - 1

    while row < m:
        while col >= 0:
            cur = matrix[row][col]
            if cur == target:
                return True
            if cur > target:
                col -= 1
            if cur < target:
                break
        row += 1
    
    return False
```

Time: O(M + N) / Space: O(1)

</details>


