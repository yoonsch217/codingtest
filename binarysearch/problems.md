### 875. Koko Eating Bananas

https://leetcode.com/problems/koko-eating-bananas

문제: piles 라는 리스트가 있고 각 원소는 바나나의 개수이다. 감시자가 h 시간동안 떠나있을 때, 시간당 k의 속도로 바나나를 먹는다. 한번에 하나의 pile만 먹을 수 있다. 전체 바나나를 다 먹기 위한 최소의 k를 구하라.

각 pile 먹는 속도는 math.ceil(pile/k) 이다. 따라서 총 걸리는 시간은 `sum of math.ceil(pile/k) for each pile in piles` 이 된다.    
전체 바나나를 다 먹을 수 있는 속도라면 possible, 못 먹으면 impossible이라고 하자.   
최대의 impossible k 보다 하나 크면 최소의 possible k가 되고, 그게 답이 된다.   
따라서 binary search로 풀 수 있다.    
left와 right를 정해야하는데 최소는 1이 되고 최대는 max(piles)가 된다.   



### 1642. Furthest Building You Can Reach

https://leetcode.com/problems/furthest-building-you-can-reach

문제: heights, bricks, ladders 가 주어진다. 건물들을 왼쪽부터 이동하는데 높은 건물로 갈 때는 높이 차이만큼 brick을 쓰든가 ladder 하나를 써야한다. 가장 멀리 갈 수 있는 건물을 찾아라.

Heap    
ladders 개수 L 을 크기로 갖는 힙을 생성하고 앞에서부터 높이 차를 넣으면서 힙을 채운다. 즉, 사다리로만 올라가되 그 높이차를 기록해놓는 것이다.   
사다리를 다 썼을 땐 이제 벽돌을 써야한다. 벽돌 써야하는 상황 왔을 때, 현재 필요한 벽돌과 힙에 있는 최소의 높이차를 비교한다.     
현재 필요한 벽돌 수가 더 적으면 벽돌 소진하면 되는 것이고, 힙에 있는 최솟값이 더 작으면 예전에 썼던 사다리를 지금 쓰고 예전 작업은 벽돌로 하면 된다.   
이렇게 해서 벽돌 수가 음수가 되면 더이상 못 가는 것이다.   

Binary Search    
특정 위치까지 갈 수 있나 없나는 판단할 수 있다. 그 구간의 height diffs를 받아서 정렬한 뒤 min부터 벽돌 사용하도록 하면 판단이 된다.    
0 ~ threshold 까지는 reachable 이고 threshold+1 ~ end 는 unreachable이다.   
이 특성을 이용해서 binary search를 사용할 수 있다.    
다만 매번 정렬을 하면 너무 cost가 크기 때문에 한번 정렬해서 height diff마다 position을 붙여놓는다. 그러고는 linear하게 iterate하면서 현재 찍은 기준 position보다 낮은 index를 가진 경우만 벽돌이나 사다리에서 차감한다.
NlogN


### 162. Find Peak Element

https://leetcode.com/problems/find-peak-element/

문제: integer array가 주어졌을 때 peak의 위치를 반환하라. peak란 주변보다 strictly greater한 값을 가지는 곳을 말하며 `nums[-1] = nums[n] = -math.inf` 으로 간주한다. O(log n) 의 알고리즘을 구하라.

left, right를 초기화하고 nums에 -math.inf 를 append한다. 이렇게 함으로써 `nums[-1] = nums[n] = -math.inf`를 자연스럽게 적용시킬 수 있다.
mid 기준에서 왼쪽이 더 크면 left half에 peak가 있어야한다. -1은 -inf이고 cur보다 cur-1이 더 크기 때문이다.
0 ~ cur-2 중에 peak가 있든가 혹은 cur-1이 peak이어야한다.   
오른쪽이 더 크면 right half에 어쨌든 peak가 있어야한다.
둘 다 아니면 현재가 peak이다.


### 658. Find K Closest Elements

https://leetcode.com/problems/find-k-closest-elements/

문제: sorted integer array가 주어지고 k, x가 주어진다. x랑 가장 가까운 k개의 원소를 정렬된 순서로 반환하라. abs(y - x)가 동일하면 작은 y값이 더 가까운 걸로 간주한다.

is_closer 짜는 게 어려웠다. 두 값이 같을 때 어디로 가야하는지를 판단해야한다.     
예를 들어 target = 5, arr = [1, 2, 2, 3, 5, 6], is_closer(1, 2)가 들어왔다면 arr[1]과 arr[2]는 모두 2로 동일하다. 하지만 2는 5보다 작으므로 arr[2]가 arr[1]보다 가깝다고 볼 수 있다. 따라서 그 값과 타겟값의 크기의 비교에 따라 정해져야한다.

<details>

```python
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        """
        Find closest first.
        Expanding left and right, find k elements.
        """
        if k == len(arr):
            return arr
        
        def is_closer(l, r):
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
                return True
            if abs(l_val - x) == abs(r_val - x):
                return l_val < r_val
            return abs(l_val - x) < abs(r_val - x)
        
        left = 0
        right = len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            
            if is_closer(mid-1, mid):
                right = mid - 1
            elif is_closer(mid+1, mid):
                left = mid + 1
            else:
                break
        # 이후는 mid 기준으로 expand. deque 대신 sliding window를 사용해서 index로만 비교하는 게 더 효율적일 거 같긴 하다. 
```

</details>

근데 솔루션 보니까 `mid = bisect_left(arr, x)` 로 간단히 구해버렸다..

그리고 신기한 솔루션도 하나 있다.   
답 subarray의 시작 지점을 찾는 것이다.   
시작 지점의 left bound는 0이 될 것이고 right bound는 n-k가 될 것이다. n-k 부터 시작을 해서 끝(n-1)까지 다 해야 k개가 되기 때문이다.   
그러면 이 두 bound에 대해 작업을 한다. left, right에 대해 mid를 구한 후 mid 값과 mid + k 값을 비교한다. mid ~ mid+k 의 subarray를 보면 크기가 k+1 이기 때문에 mid나 mid+k 중 하나는 버려져야한다. k+1로 하는 이유는 최적화된 subarray가 더 오른쪽에 있는지 왼쪽에 있는지 알아야하기 때문이다.      
mid가 target에 더 가깝다면 mid+k 이후는 subarray에 포함될 수가 없으므로 버려야하고 subarray의 시작지점은 mid 이하가 된다. 안 그러면 mid를 버려야하는데 mid가 더 가깝기 때문에 버리면 안 되기 때문이다.   
right(시작지점의 right bount)를 mid로 옮기고 다음 iteration을 진행한다.   
이걸 반복하다가 left == right 되는 순간이 답이다.


### 744. Find Smallest Letter Greater Than Target

https://leetcode.com/problems/find-smallest-letter-greater-than-target/

문제: 알파벳 소문자로 이루어진 리스트가 주어지고 target character가 주어진다. target char보다 큰 문자 중 가장 작은 문자를 반환하라. target char보다 큰 게 없으면 첫 번째 문자를 반환하라.

`while left <= right` 로 binary search를 먼저 했다. target을 찾는 게 아니라 target보다 큰 최소를 찾는 거니까 `letters[mid] <= target` 이면 left를 옮기고 아니면 right를 옮기도록 했다.   
while이 끝났을 때 left가 len 이상이면 못 찾고 끝난 거니까 letters[0]을 반환한다. 아니면 letters[left]를 반환한다. left가 condition을 만족하는 가장 작은 값이다.   



### 153. Find Minimum in Rotated Sorted Array

https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/

문제: sorted array with unique integer가 주어진다. 한 군데를 기준으로 rotate 되어 있는데 그 array의 최솟값을 구하라. `ex) [3, 4, 5, 1, 2] ` 

start < end면 rotated가 안 된거니까 그거에 대한 처리를 먼저 한다. rotated 된 경우에 대해서는 index 0의 value를 기준으로 잡는다.   
`index 1 ~ rotated point`: index 0의 값보다 큰 값이다.   
`rotated point ~ last`: index 0의 값보다 작은 값이다.    
binary search를 써서 mid가 nums[0]보다 작았을 때, nums[mid-1]이 nums[0]보다 크거나 같으면 rotate 직후의 값이므로 nums[mid]를 반환한다. 아니면 left half를 탐색한다(right를 옮긴다). 이 때, mid는 항상 1보다 크므로 별도의 range check을 할 필요는 없다.   
mid가 nums[0]보다 크면 right half를 탐색한다(left를 옮긴다).    




### 154. Find Minimum in Rotated Sorted Array II

https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/

문제: 중복된 값이 있는 integer array가 ascending order로 정렬되어 있었는데 몇 번의 rotation이 일어난 상태이다. 이 array에서 최솟값을 구하라.

(내 풀이) 중복된 값이 있음으로서 달라지는 부분이 있다.   
로직은 다음과 같다.   
- 기본 binary search 템플릿을 사용한다.
- nums[left] < nums[right] 라면 그 subarray는 rotation이 없는 것이므로 nums[left] 를 반환한다.
- mid가 mid-1보다 작으면 mid가 rotation point이므로 nums[mid]를 반환한다. left가 최솟값이 아님을 보장하므로 mid-1은 항상 존재한다. 이 다음은 항상 rotated되었음을 가정하고 진행할 수 있다.
- mid가 left보다 작다면 rotation point가 왼쪽에 있을테니 left half를 탐색한다.
- mid가 left보다 크다면 rotation point가 오른쪽일테니 right half를 탐색한다. 
- mid가 left와 같을 때가 문제이다.
  - right가 left보다 작다면 right half를 탐색한다. right가 left보다 큰 건 이미 처리했다.
  - right가 left와 같을 때만 남았는데 이 때는 linear하게 탐색을 한다.

(solution) 비슷한데 훨씬 간단하다. right랑 비교하면 unrotated array에 대해 고려하지 않아도 되네? 왜지,
- mid가 right보다 작으면 left half를 본다. left에서 mid 사이에 rotation point가 있어야한다.
- mid가 right보다 크면 right half를 본다.
- 아니면 right를 하나 줄임으로써 범위를 좁힌다.
