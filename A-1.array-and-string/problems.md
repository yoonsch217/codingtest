### 3. Longest Substring Without Repeating Characters

https://leetcode.com/problems/longest-substring-without-repeating-characters

문제: 문자열이 주어졌을 때 반복되는 글자가 없는 가장 긴 substring의 길이를 반환하라.

<details><summary>Approach 1</summary>

sliding window를 사용한다. substring의 left와 right를 정해주는 포인터 두 개를 저장한다.   
그리고 dictionary 하나를 만들어서 key는 나타났던 문자, value는 그 문자의 위치를 저장한다.   
string을 traverse하면서 현재 문자가 dictionary에 있으면서 그 문자의 위치가 cur_idx보다 크거나 같으면 left pointer를 `d[cur] + 1` 로 업데이트한다.   
그리고 `d[cur] = right` 로 업데이트 혹은 추가를 해주고 `right += 1` 로 포인터 위치를 옮긴다.    

Time: O(N) , Space: O(N)



```python
def lengthOfLongestSubstring(self, s: str) -> int:
    l = 0
    char_to_idx = {}
    ans = 0
    for r in range(len(s)):
        cur = s[r]
        if cur in char_to_idx:
            l = max(char_to_idx[cur] + 1, l)  # 이 부분 조심! l = r 로 했다가 틀렸다. l = char_to_idx[cur]+1 로 해야한다.
        char_to_idx[cur] = r
        ans = max(ans, r - l + 1)
    return ans
```

</details>



### 1094. Car Pooling

https://leetcode.com/problems/car-pooling/

문제: capacity 라는 integer와 passengers라는 리스트가 주어진다. passengers 리스트는 `[승객수, 탑승 시각, 하차 시각]` 로 구성되어 있다. 전체 여정을 하면서 capacity를 넘는 순간이 있다면 false, 그렇지 않다면 true를 반환한다.


<details><summary>Approach 1</summary>

1. Sum get-on/get-off passengers for each location
2. Sort by time
3. Put two pointers to each list, and iterate from the start. If exceeds capacity, return False

   
`dd = sorted(d.items())` 이렇게 하면 dict가 key로 정렬된 후에 tuple list로 저장이 된다. d.items() 자체가 tuple로 변환된 것이다.   

한 location에 내리는 trip이 여러 개 있을 때, 이걸 처음에 froms, tos 만들 때 한 location에 합쳐서 데이터를 만들어야 한다.   
근데 dict 대신 tuple list를 사용하면 (location, disembarking1), (location, disembarking2) 이렇게 두 개가 만들어지게 된다.   
그러면 나중에 sorting 했을 때 각 iteration에서 하나씩 계산하면 안 되고 동일 location이 아닐 때까지 뒤로 탐색을 더 해야해서 불편하다.   

sort 작업 때문에 Time: O(N log N), Space: O(N) 일 것 같다.


```python
def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
    getons = defaultdict(int)
    getoffs = defaultdict(int)
    for _num, _from, _to in trips:
        getons[_from] += _num
        getoffs[_to] += _num
    
    getons = sorted(getons.items(), key=lambda x: x[0])
    getoffs = sorted(getoffs.items(), key=lambda x: x[0])

    i_on = i_off = 0
    cur_cnt = 0

    while i_on < len(getons) and i_off < len(getoffs):
        geton_loc = getons[i_on][0]
        getoff_loc = getoffs[i_off][0]
        if geton_loc <= getoff_loc:
            cur_cnt += getons[i_on][1]
            i_on += 1
        if geton_loc >= getoff_loc:
            cur_cnt -= getoffs[i_off][1]
            i_off += 1
        if cur_cnt > capacity:
            return False
    return True
    
```

</details>




<details><summary>Approach 2</summary>

조건을 보니까 시각이 1 ~ 1000까지라는 제약이 있다. bucket sort를 사용할 수도 있다.    
Time: O(N), Space: O(1)


```python
boardings = [0] * 1001
disembarkings = [0] * 1001
for _num, _from, _to in trips:
    boardings[_from] += _num
    disembarkings[_to] += _num

sum_passengers = 0
for i in range(1001):
    sum_passengers += (boardings[i] - disembarkings[i])
    if sum_passengers > capacity:
        return False
return True
```

흠 근데 시간이 더 안 빨라지네.

</details>






### 2158. Amount of New Area Painted Each Day

https://leetcode.com/problems/amount-of-new-area-painted-each-day/  (locked)

문제: 0-indexed 2D integer array paint가 주어진다. paint[i] = [start_i, end_i] 인데 i 시점에 start position과 end position 사이를 칠할 수 있다. paint array를 앞에서부터 iterate하면서 해당 범위 내에 색칠을 하는데 같은 부분을 두 번 이상 색칠하지는 않을 때 각 iteration에서 색칠해아 하는 칸의 수를 구하라.   
더 앞에 있는 날이 칠한 곳은 뒤에 있는 날이 칠할 수 없다.   


예시
- 입력: paint = [[1, 4], [4, 7], [5, 8]]
- 출력: [3, 3, 1]
  - 1일차: [1, 4] 영역 색칠 (새 영역: 3)
  - 2일차: [4, 7] 영역 색칠 (새 영역: 3)
  - 3일차: [5, 8] 영역 중 [5, 7]은 이미 2일차에 칠해짐 → [7, 8]만 새로 칠함 (새 영역: 1)


<details><summary>Approach 1</summary>

sweep line이라는 개념이 들어간다.   
https://leetcode.com/problems/amount-of-new-area-painted-each-day/discuss/1740812/Python-Complete-3-solutions-using-different-data-structures

- start position과 index를 tuple로 묶어서 position array에 넣는다. end position과 index도 마찬가지로 넣는다. start인지 end인지 알 수 있도록 넣는다.
- position을 기준으로 sort를 한다. 리스트를 앞에서부터 스캔하면 빠른 position부터 나올 것이다.   
- 전체 길이에 맞는 buffer array를 만든다.   
- position array를 iterate하면서 start가 나오면 그때부터 buffer의 해당 position에 해당 index를 넣는다. end가 나오면 해당 index에 대해서 그만 넣는다.   
- 다 끝나고 buffer array를 살펴봤을 때 어떤 칸에 1, 5가 있다면 인덱스 1 작업과 인덱스 5 작업에 의해 색칠될 수 있던 공간이라는 뜻이다.   
인덱스 낮은 작업이 우선이므로 인덱스 1의 작업에 의한 색칠 부분으로 인식하면 된다.   

직관적인 것 같다. 실제로 paint array 를 iterate 하면서 칠한다고 생각해보자. 먼저 칠하고, 그 다음 칠할 때는 칠해야하는 부분 중에서 하얀 부분만 칠하는 것이다.

```py
def amountPainted(paint):
    # 1. position array 만들기 (start/end 지점 기록)
    # 어느 좌표에서 어떤 index가 시작되고 끝나는지 저장
    starts = defaultdict(list)
    ends = defaultdict(list)
    
    max_pos = 0
    for i, (s, e) in enumerate(paint):
        starts[s].append(i)
        ends[e].append(i)
        max_pos = max(max_pos, e)
        
    # 2. 전체 길이에 맞는 buffer array (각 칸마다 어떤 index들이 겹치는지 저장)
    # buffer[pos] = [idx1, idx2, ...]
    buffer = [[] for _ in range(max_pos + 1)]
    
    # 3. 이벤트를 훑으며 buffer 채우기
    current_indices = set()
    # 좌표 0부터 max_pos까지 순회 (Single Loop)
    for pos in range(max_pos + 1):
        # 현재 좌표에서 새로 시작되는 작업 추가
        if pos in starts:
            for idx in starts[pos]:
                current_indices.add(idx)
        
        # 현재 좌표에서 종료되는 작업 제거
        if pos in ends:
            for idx in ends[pos]:
                current_indices.remove(idx)
                
        # 현재 칸(pos ~ pos+1)에 유효한 인덱스들을 복사해서 저장
        if current_indices:
            buffer[pos] = list(current_indices)

    # 4. buffer를 살펴보고 가장 낮은 인덱스의 작업으로 인식
    res = [0] * len(paint)
    for pos in range(max_pos):
        if buffer[pos]:
            # 인덱스 낮은 작업이 우선권
            winner_idx = min(buffer[pos])
            res[winner_idx] += 1
            
    return res
```

</details>


### 904. Fruit Into Baskets

https://leetcode.com/problems/fruit-into-baskets/description/

문제: fruits 라는 리스트가 주어지고 각 index에 있는 값은 그 위치에 있는 과일을 의미한다. 과일이 리스트대로 일렬로 나열되어 있고 사용자는 어느 한 지점부터 오른쪽으로 과일을 주워담는다. 최대 두 종류의 과일까지 담을 수 있고 그걸 넘어서는 순간 담을 수 없고 멈춰야한다. 최대로 많이 담을 수 있는 과일의 수를 구하라.

<details><summary>Approach 1</summary>

related topic을 보니까 sliding window가 나와서 그 방법으로 풀었다.   

- 최대 길이가 2인 dictionary를 만들고 key는 fruit, value는 그 fruit이 지금까지 나온 위치 중 가장 오른쪽 위치를 저장한다. 
- left, right 포인터를 두고 right 포인터를 하나씩 오른쪽으로 옮긴다. 
- dict에 없는 3번째의 과일이 나오게 되면 left 포인터를 옮긴다. 이 때 dict에 있는 두 가지 과일 중 더 왼쪽에 있는 과일을 버려야한다. 그 과일의 위치 바로 다음부터가 동일한 과일이 연속으로 나오기 시작한 위치이기 때문에 거기에 left 포인터를 놓고 현재 right의 과일을 dictionary에 추가한다.


  
```python
def totalFruit(self, fruits: List[int]) -> int:
    n = len(fruits)
    d = {}  # key: fruit, value: rightmost index of the fruit
    res = 0

    left = right = 0  # fruits[left:right+1] 까지를 대상으로 한다. right is the current pointer

    while right < n:
        cur = fruits[right]
        if cur not in d and len(d) >= 2:
            # Move left pointer to the position where the only very previous fruit started to appear before the current fruit
            fruit_to_drop = min(d, key=d.get)  # 깔끔하네. 나는 items()로 펼친 다음에 if 두 개 써서 구했는데.
            left = d[fruit_to_drop] + 1  # This is the position where the other fruit starts to appear consecutively
            del d[fruit_to_drop]
        d[cur] = right
        res = max(res, right - left + 1)
        right += 1

    return res
```

</details>






### 128. Longest Consecutive Sequence

https://leetcode.com/problems/longest-consecutive-sequence/description/

문제: Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
 You must write an algorithm that runs in O(n) time.
 `[100,4,200,1,3,2]` => 4


<details><summary>Approach 1</summary>

set 사용

- 모든 값들을 set에 넣는다.
- nums를 iterate하면서 현재 num에서 +1 씩 올려가면서 nums set에 존재하는지 확인한다.
- nums set 에 없을 때가 도달하면 answer를 업데이트하고 다음 nums 를 iterate 한다.
- n에 대한 iteration에서 num-1 이 nums set 에 있으면 그 iteration은 건너뛴다. num-1 에 대한 iteration 에서 처리될 것이기 때문이다.
- set 만드는 데 O(N), iteration 동안 최대 2N 번 접근하니까 O(2N) 이다. 총 O(3N) = O(N)


```py
    def longestConsecutive(self, nums: List[int]) -> int:
        num_set = set(nums)
        best = 0
        for num in num_set:  # for num in nums로 하면 느려진다. 중복이 제거가 안 되니까!
            if num - 1 in num_set:
                continue
            tmp = num + 1
            cnt = 1
            while tmp in num_set:
                tmp += 1
                cnt += 1
            best = max(best, cnt)
        return best
        
```

</details>

<details><summary>Approach 2</summary>

Hash Map: 이거 좀 까다롭다.

개념
- Hash Map: 해당 num 에서의 연속된 숫자의 최대 길이
- Bridge Building: 새로운 숫자 num이 들어올 때, 이미 존재하는 왼쪽 그룹(num-1)과 오른쪽 그룹(num+1)을 하나로 잇는 다리 역할을 수행한다.
- Boundary Update: 전체 그룹의 길이를 모든 원소에 기록할 필요 없이, 그룹의 양 끝단에만 최신 길이를 기록한다. 중간에 낀 숫자들은 다시 조회되지 않는다.

동작
- 각 위치에 대해서 최대로 expand할 수 있는 길이를 저장하는 hash map을 사용한다.   
- 어떤 값 num에 대해 
   - `d[num]`이 존재한다면 이미 처리한 적 있는 값이므로 넘어간다.
   - `d[num-1]`이  존재한다면 num-1로부터 왼쪽으로 `d[num-1]` 만큼 값이 있다는 것이다. 없다면 left로는 0만큼 expand할 수 있다.
     - 왼쪽으로만 연결되어 있는 이유는, num 값을 이 알고리즘이 처리한 적이 없기 때문에 이 존재를 모른 상태로 최댓값을 구한 것이기 때문이다. 
   - `d[num+1]`이  존재한다면 num+1로부터 왼쪽으로 `d[num+1]` 만큼 값이 있다는 것이다. 있다면 right로는 0만큼 expand할 수 있다.
   - `d[num] = left + right + 1`이 되고 `ans = max(ans, d[num])`이 된다.
   - `d[num-left]` 값도 `d[num]` 값과 동일하게 된다. num-left부터 오른쪽으로 `d[num]`만큼 확장시킬 수 있다.
    이제 `d[num-left]`를 사용할 값은 num-left-1이 된다.
- 좌우로 expand 할 때 하나 씩 차이가 나면서 expand 하기 때문에 가능하다.

복잡도
- 시간: O(N)
- 공간: O(N)


```py
    def longestConsecutive(self, nums: List[int]) -> int:
        ans = 0
        d = {}

        for num in nums:
            if num not in d:
                left = d[num-1] if num-1 in d else 0
                right = d[num+1] if num+1 in d else 0
                cur_len = left + right + 1
                d[num] = cur_len
                ans = max(ans, cur_len)

                d[num-left] = cur_len
                d[num+right] = cur_len
            else:
                continue
        
        return ans

```

</details>










### 75. Sort Colors

https://leetcode.com/problems/sort-colors

문제: Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.
We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.
You must solve this problem without using the library's sort function.

<details><summary>Approach 1</summary>


selection sort를 사용하면 된다.   
맨 앞 element 부터 차례대로, 오른쪽으로 iterate하면서 최솟값과 swap을 한다. O(N^2) / O(1)

```py
def sortColors(self, nums):
    for i in range(len(nums)):
        min_val = nums[i]
        min_idx = i
        for j in range(i, len(nums)):
            if nums[j] < min_val:
                min_val = nums[j]
                min_idx = j
        nums[i], nums[min_idx] = nums[min_idx], nums[i]
```

아니면 각각 count를 세서 앞에서부터 채워도 된다.   
근데 in-place를 원하는데 count를 사용하면 별도의 메모리를 사용하는 거라 별로일 것 같다. 그래도 key 개수가 정해져있으니 O(1) 메모리이긴 하네. 
O(N) / O(1)


```py
def sortColors(self, nums: List[int]) -> None:
    d = defaultdict(int)
    for num in nums:
        d[num] += 1
    
    ptr = 0
    keys = [0, 1, 2]
    for key in keys:
        for i in range(d[key]):
            nums[ptr] = key
            ptr += 1
```

</details>


<details><summary>Approach 2</summary>

Dutch National Flag

one pass algorithm이 있다.   
올바른 결과에서 0은 왼쪽부터, 2는 오른쪽부터 채워지면 되고 1은 나머지에 있으면 된다.   

- p0을 제일 왼쪽, p2를 제일 오른쪽으로 둔다.   
- cur 라는 포인터를 왼쪽부터 iterate하면서 0이면 p0과 swap하고 p0과 cur 한 칸 올린다.   
- 2면 p2와 swap 후 p2 한 칸 내리기, 1이면 skip 하고 cur 올리면 된다.   
- 0일 땐 p0와 cur를 둘 다 올리는 게 중요하다.


```py
    def sortColors(self, nums: List[int]) -> None:
        n = len(nums)
        zero_ptr, cur, two_ptr = 0, 0, n-1
        # all on the left of zero_ptr are zeros 
        # all on the right of two_ptr are twos 

        while cur <= two_ptr:  # 이 조건을 잘 넣어줘야한다. cur < len(nums) 하면 마지막에 2랑 1이 다 바뀌어버린다. cur < two_ptr 하면 하나가 정렬이 안 된다.
            num = nums[cur]
            if num == 0:
                nums[zero_ptr], nums[cur] = num, nums[zero_ptr]
                zero_ptr += 1
                cur += 1  # 여기서 cur를 안 올리면 infinite loop 에 빠질 수 있다.
            elif num == 2:
                nums[two_ptr], nums[cur] = num, nums[two_ptr]
                two_ptr -= 1  # 여기는 cur 를 올리면 안 된다! 뒤에서 온 게 뭔지 모르니까. zero_ptr 인 경우는 cur를 올리는 이유가, zero_ptr 에 있던 값이 0이거나 1이기 때문이다.
            else:
                cur += 1
```

</details>


### 15. 3Sum

문제: Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

Example 1:
- Input: nums = [-1,0,1,2,-1,-4]
- Output: [[-1,-1,2],[-1,0,1]]
- Explanation: 
  - nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
  - nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
  - nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
  - The distinct triplets are [-1,0,1] and [-1,-1,2].
  - Notice that the order of the output and the order of the triplets does not matter.


<details><summary>Solution</summary>

Two pointers

헤맨 포인트
- 처음에 중복된 답도 찾아야하는 줄알고 pointer 옮기는 게 어려웠다. 근데 중복된 답 제거하는 것이기 때문에 ans 를 찾았을 때 left, right 둘 다 옮기면 된다.
- 중복을 제거하는 게 헷갈렸다. start index 기준에서도 이전과 동일한 값이라면 건너뛰어야하고, left 와 right 기준에서도 이전과 동일한 값을 다 건너뛰도록 해야한다.

```python
class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        """
        nums = [-1, 0, 1, 2, -1, -4]
        output = [[-1, -1, 2]. [-1, 0, -1]]

        Approach 1:
        brute force: Get all the combinations with three different numbers, and add to the output if it sums up to zero
        Time: (N^3)

        Approach 2
        Get the sum of every two numbers, and if the sum is k, find -k with binary search
        Time: (N^2 * logN)

        Approach 3
        Sort and use pointers
        nums = [-4, -3, -2, -2, -1, -1, 0, 1, 3, 5, 6]
        Set left and right pointers
        For each left pointer, iterate right pointer from the rightmost to left, getting all the possible answers with [left, right_iter, mid_iter]. We can find mid_iter with bs.
        Repeat until left is not over 0.
        
        N^2 * log N

        Approach 4
        Sort and use two pointers
        Fix left pointer, and find all the pairs of numbers that make up to (-nums[left_pointer])
        When answer found, move both left and right, because we don't need duplicate answers.
        O(N^2)
        """
        nums.sort()
        start = 0

        res = []

        while start < len(nums) and nums[start] <= 0:
            if start > 0 and nums[start-1] == nums[start]:
                start += 1
                continue

            target = -nums[start]
            left = start + 1
            right = len(nums) - 1
            while left < right:

                if nums[left] + nums[right] == target:
                    res.append([nums[start], nums[left], nums[right]])
                    left += 1
                    right -= 1

                    # 중복 제거
                    while left < right and nums[left] == nums[left-1]:
                        left += 1
                    while left < right and nums[right] == nums[right+1]:
                        right -= 1

                elif nums[left] + nums[right] < target:
                    left += 1
                else:
                    right -= 1
            start += 1
        
        return res
```

</details>




### 56. Merge Intervals

문제: Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, 
and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:
- Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
- Output: [[1,6],[8,10],[15,18]]
- Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].



<details><summary>Solution</summary>

처음에 union find 기법으로 풀었다. 
근데 비효율적이다. O(N^2) 의 시간이 걸리기도 하고, union 및 find 작업이 자주 일어나는 상황이 아니라 처음 그룹화에만 필요했다.

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        roots = [i for i in range(len(intervals))]

        def find(idx):
            if idx == roots[idx]:
                return idx
            roots[idx] = find(roots[idx])
            return roots[idx]
        
        def union(idx1, idx2):
            root1 = find(idx1)
            root2 = find(idx2)
            if root1 == root2:
                return
            roots[root1] = root2
            return

        for i, interval in enumerate(intervals):
            start_1, end_1 = interval
            for j in range(i+1, len(intervals)):
                start_2, end_2 = intervals[j]
                if (end_1 >= start_2 and end_2 >= end_1) or (end_2 >= start_1 and end_1 >= end_2):
                    union(i, j)
        
        root_to_range = {}
        for i, interval in enumerate(intervals):
            root = find(i)
            if root not in root_to_range:
                root_to_range[root] = interval
            else:
                prev_start, prev_end = root_to_range[root]
                cur_start, cur_end = interval
                root_to_range[root] = [min(prev_start, cur_start), max(prev_end, cur_end)]

        return list(root_to_range.values())
```

더 간편한 방법이 있다.: sort + greedy
- start 기준으로 정렬을 한다.
- 앞에서부터 iterate 하면서 바로 뒷 범위와 overlap 이 있는지 확인한다.
- overlap 이 있으면 업데이트를 하면서 결과 리스트를 만든다.
- Complexity
  - Time: O(N logN) 정렬
  - Space: O(1)

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda x: x[0])
        
        res = []
        cur = intervals[0]
        for i in range(1, len(intervals)):
            cur_start, cur_end = cur  # 여기가 cur 를 참조해야한다. 처음에는 intervals[i] 와 intervals[i+1] 를 사용했는데 그러면 지금까지 만들어놓은 리스트를 기준으로 비교할 수가 없다.
            next_start, next_end = intervals[i]
            # check overlap
            if cur_end >= next_start:
                cur = [cur[0], max(next_end, cur_end)]
            else:
                res.append(cur)
                cur = intervals[i]
        res.append(cur)
        return res
```


</details>




### 347. Top K Frequent Elements

문제: Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

Example 1:
- Input: nums = [1,1,1,2,2,3], k = 2
- Output: [1,2]


<details><summary>Solution</summary>

heap 방식은 간단하다. O(N logN) 의 시간 복잡도를 갖는다.

아래와 같이 bucket 을 사용한 방법도 있다. 최적일 때 O(N) 의 시간이 걸린다. 

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        max_freq = 0
        num_to_frequency = defaultdict(int)
        for num in nums:
            prev = num_to_frequency[num]
            num_to_frequency[num] = prev + 1
            max_freq = max(max_freq, prev + 1)
        
        count_to_nums = [[] for _ in range(max_freq+1)]
        for num, freq in num_to_frequency.items():
            count_to_nums[freq].append(num)

        res = []
        count_idx = len(count_to_nums) - 1
        filled = 0
        while filled < k:
            cur_nums = count_to_nums[count_idx]
            if len(cur_nums) == 0:
                count_idx -= 1
                continue
            if len(cur_nums) + filled <= k:
                res.extend(cur_nums)
                filled += len(cur_nums)
            else:
                res.extend(cur_nums[:k-filled])
                filled += (k-filled)
            count_idx -= 1
        
        return res
```

</details>


### 76. Minimum Window Substring

문제: Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. 
If there is no such substring, return the empty string "".
The testcases will be generated such that the answer is unique.

- Input: s = "ADOBECODEBANC", t = "ABC"
- Output: "BANC"
- Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

<details><summary>Solution</summary>

- hard 문제 치고는 접근이 쉬웠다.
- counter map 기준으로 비교를 한다.
- t에 대한 counter map 을 만든다.
- s에 대해서 left, right 포인터를 움직이면서 t의 counter map 을 충족하는 window를 찾는다. 
  - 충족하는지는 num_found 라는 변수를 사용했다. 각 char 마다 개수를 충족하는 char가 생길 때마다 num_found를 높였다. 
- 이 window에서 left를 줄여가면서 counter map을 충족하는 최소 크기를 찾는다.
  - 조건을 만족하는 window 이므로 res를 업데이트를 한다. 
  - left 위치에 있는 char 에 대해 counter map 을 줄인다. 이때 해당 char의 개수가 target 보다 작아지게 된다면 num_found 값도 업데이트한다.
- 조건을 만족하지 않는 window가 만들어지는 경우, 이제 다시 right pointer를 옮겨가며 조건에 맞는 window를 찾는다.

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(s) < len(t):
            return ""
        res = (math.inf, 0, 0)  # length, left index, right index
        l, r = 0, 0
        char_to_count_s = defaultdict(int)
        char_to_count_t = Counter(t)

        num_found = 0  # compared with len(char_to_count_t)
        num_found_target = len(char_to_count_t)

        while r < len(s):
            cur = s[r]
            char_to_count_s[cur] += 1
            if cur in char_to_count_t and char_to_count_s[cur] == char_to_count_t[cur]:
                num_found += 1
            
            while num_found == num_found_target and l <= r:
                cur_len = r - l + 1
                if cur_len < res[0]:
                    res = (cur_len, l, r)
                left_char = s[l]
                char_to_count_s[left_char] -= 1
                if left_char in char_to_count_t and char_to_count_s[left_char] < char_to_count_t[left_char]:
                    num_found -= 1
                l += 1
            
            r += 1
        
        if res[0] == math.inf:
            return ""
        return s[res[1]: res[2]+1]
```

</details>



### 215. Kth Largest Element in an Array

문제: Given an integer array nums and an integer k, return the kth largest element in the array.
Note that it is the kth largest element in the sorted order, not the kth distinct element.
Can you solve it without sorting?

- Input: nums = [3,2,1,5,6,4], k = 2
- Output: 5

<details><summary>Solution</summary>

- quick select 방식을 먼저 써봤다.
- quick sort의 원리를 활용한다. pivot을 무작위로 고르고 그것보다 작은 값은 앞에, 큰 값은 뒤에 몰아넣으면 pivot 에 해당하는 숫자는 자기 자리를 찾게 된다.
- 이 자리가 문제에서 찾는 위치라면 그 값을 반환하고, 아니라면 위치에 따라 왼쪽을 볼지 오른쪽을 볼지 결정해서 더 본다.
- 이론적으로는 average O(N) 이지만 TLE 발생한다.

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        target_idx = len(nums) - k
        left, right = 0, len(nums) - 1
        while left <= right:
            pivot = random.randint(left, right)
            target = nums[pivot]
            nums[pivot], nums[right] = nums[right], nums[pivot]  # pivot number moved to right end
            low, high = left, right-1
            
            while low <= high:
                if nums[low] >= target:
                    nums[low], nums[high] = nums[high], nums[low]
                    high -= 1
                else:
                    low += 1

            nums[low], nums[right] = nums[right], nums[low]
            if low == target_idx:
                return nums[low]
            if low < target_idx:
                left = low + 1
            else:
                right = low - 1
```

TLE가 발생했던 이유는 pivot value와 동일한 값이 많을 때 scope이 안 줄여지기 때문이다. 그래서 Dutch National Flag 기법을 활용해서 세 가지로 나눈다.

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        target_idx = len(nums) - k
        left, right = 0, len(nums) - 1
        
        while left <= right:
            pivot_idx = random.randint(left, right)
            pivot_val = nums[pivot_idx]
            
            # 3-way partition (Dutch National Flag Algorithm)
            # lt: 피벗보다 작은 값들이 끝나는 지점
            # gt: 피벗보다 큰 값들이 시작되는 지점
            # i: 탐색 포인터
            lt = left
            gt = right
            i = left
            
            while i <= gt:
                if nums[i] < pivot_val:
                    nums[lt], nums[i] = nums[i], nums[lt]
                    lt += 1
                    i += 1
                elif nums[i] > pivot_val:
                    nums[gt], nums[i] = nums[i], nums[gt]
                    gt -= 1
                else:
                    i += 1
            
            # 이제 [lt, gt] 구간은 모두 피벗과 같은 값들입니다.
            if lt <= target_idx <= gt:
                return nums[lt]
            elif target_idx < lt:
                right = lt - 1
            else:
                left = gt + 1
```

</details>



### 57. Insert Interval

문제: You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by start. 
You are also given an interval newInterval = [start, end] that represents the start and end of another interval.
Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).
Return intervals after the insertion.

- Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
- Output: [[1,5],[6,9]]

<details><summary>Solution</summary>

- brute force approach
  - 처음에는 intervals 를 iterate 하면서 각각 new_interval과 비교하고, is_overlapping 플래그를 사용하려고 했다.
  - 이렇게 하니까 edge case가 너무 많이 생겨서 복잡했다. interval 과 겹치는 것 없이 맨 앞에 있는 경우, 맨 뒤에 있는 경우 등등
- merge 할 때는 세 가지로 나눈다: 겹치는 거 생기기 전, 겹치는 중, 겹치는 거 끝난 후

```python
class Solution:
    def insert(self, intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
        res = []
        cur_idx = 0
        n = len(intervals)
        new_start, new_end = new_interval

        # before overlapping
        while cur_idx < n:
            cur_start, cur_end = intervals[cur_idx]
            if cur_end >= new_start:
                break
            res.append([cur_start, cur_end])
            cur_idx += 1
        
        # during overlapping
        overlap_start, overlap_end = new_start, new_end
        while cur_idx < n:
            cur_start, cur_end = intervals[cur_idx]
            if cur_start > new_end:
                break
            overlap_start = min(overlap_start, cur_start)
            overlap_end = max(overlap_end, cur_end)
            cur_idx += 1
        res.append([overlap_start, overlap_end])
        
        # after overlapping
        while cur_idx < n:
            res.append(intervals[cur_idx])
            cur_idx += 1
        
        return res
```


</details>





### Design Hit Counter

- 문제: 당신은 지난 5분(300초) 동안 발생한 '히트(클릭 등)'의 수를 기록하는 시스템을 설계해야 합니다.
- 요구 사항
  - hit(timestamp): 타임스탬프 timestamp(초 단위)에 히트가 발생했음을 기록합니다. 
  - 입력되는 timestamp는 시간에 따라 **단조 증가(monotonically increasing)**한다고 가정합니다 (즉, 1, 2, 5, 10... 순으로 들어옴). 
  - 여러 히트가 동일한 타임스탬프에 발생할 수 있습니다. 
  - getHits(timestamp): 현재 타임스탬프 timestamp를 기준으로, 최근 5분(즉, 300초) 동안 발생한 총 히트 수를 반환합니다. 
  - 범위 계산 시 현재 타임스탬프를 포함합니다: [timestamp - 299, timestamp].

```
counter = HitCounter()
counter.hit(1)       # 1초에 히트 발생
counter.hit(2)       # 2초에 히트 발생
counter.hit(3)       # 3초에 히트 발생
counter.getHits(4)   # 4초 기준 최근 300초(1~4초) 내 히트 수 -> 3 반환
counter.hit(300)     # 300초에 히트 발생
counter.getHits(300) # 300초 기준 최근 300초(1~300초) 내 히트 수 -> 4 반환
counter.getHits(301) # 301초 기준 최근 300초(2~301초) 내 히트 수 -> 3 반환 (1초의 히트 제외)
```


<details><summary>Solution</summary>

- 요구사항
  - hit 를 저장 => 메모리 필요
  - hit 수를 받아오기
    - 받아올 시간 주어짐 => 모든 hit history 저장 필요 & 원하는 시간의 데이터 위치를 빠르게 찾아야 함
    - 받아올 범위가 주어짐 => hit history 가 연속적으로 저장되어 있어야 함 (range가 고정되어 있음. 이걸 미리 알고 저장하면 최적화 가능?)
- approach 1
  - hit 를 list 에 (timestamp, counter) 로 저장 O(1)
  - hit를 받아올 때 timestamp 기준으로 binary search 후 앞에 300만큼 탐색 O(log N)
- approach 2
  - 최근 300 기간 동안의 값만 저장하는 queue 를 갖는다. [timestamp, counter]
  - 최근 300 기간의 counter sum 을 저장하는 변수를 갖는다. `recent_sum`
  - 각 timestamp 마다 최근 300 기간의 sum을 유지하는 hashmap을 갖는다. key: timestamp, value: sum of hits
  - 새로운 hit이 들어올 때
    - queue 맨 뒤를 확인하고 새로 추가하든가 기존의 값을 증가시킨다. 이걸 바탕으로 recent_sum 을 업데이트한다.
    - queue 맨 앞을 확인하고 범위를 벗어난 값들은 버린다. 이걸 바탕으로 recent_sum을 업데이트한다.
    - 이렇게 결정된 recent_sum 을 hashmap에 저장을 한다. map[timestamp] = recent_sum
  - hit 조회를 할 때
    - 주어진 timestamp 에 대한 value만 받아오면 된다.
    - 해당 timestamp 가 hashmap key 에 없을 때, 이게 안 되네 => 사용 불가
  - Complexity
    - Time: hit O(1), get_hit O(1)
    - Space: O(N)

내가 문제를 잘 이해 못 했던 게 있다. get_hits 로 주어지는 timestamp 와 hit로 주어지는 timestamp 모두 단조 증가인 것이다. 즉, 모든 history를 저장할 필요가 없다.

- approach 3
  - `hit_buffer` 최근 300 기간 동안의 값을 저장하는 queue 를 갖는다. [timestamp, count for current timestamp, sum of counts over the last 300 seconds]
  - hit 들어올 때
    - hit_buffer 의 가장 최근 timestamp 와 같다면 거길 업데이트한다.
    - 그렇지 않다면, new timestamp 가 될 때까지 hit_buffer 를 업데이트한다. 비어있는 시간도 업데이트해야한다. 
      - latest timestamp 보다 new timestamp - 299 이 더 크면 이것부터 시작을 한다.
      - latest timestamp 에 대한 count sum 을 받아온 뒤, 앞에서 pop 을 할 때마다 그 count를 빼줘야한다.
  - hit 조회할 때
    - 가장 최근의 timestamp 와 조회할 timestamp 를 알면 원하는 timestamp 의 index를 바로 알 수 있다. 거기에 저장된 값을 반환한다.
  - Complexity
    - Time: hit O(1), get O(1)
    - Space: O(300) = O(1)
  - 구현 어렵네 이거.

코너 케이스
- 코너 케이스 고려
  - timestamp의 범위는?
  - get_hits 에서 주어진 timestamp 의 범위는? 음수가 들어올 때는? exception 처리하자.


Using fixed size array   
Approach 1
- fixed size array를 사용한다. 길이 300자리 array를 사용한다. 
  - 각 array에는 [timestamp, count sum] 이 저장되어 있다.
- hit 처리
  - (new timestamp % 300) 의 위치를 업데이트해야한다. 
    - 기존에 timestamp 가 동일하다면 count sum 만 하나 증가시킨다.
    - 다르다면, 전체를 scan 하면서 timestamp - 299 의 범위 중에서 가장 최신의 값을 받아온 뒤 거기서 하나 증가시킨다.
- get hit 처리
  - (timestamp % 300) 의 위치의 timestamp 값이 입력 timestamp 와 같다면 그 값을 반환한다.
  - 다르다면.. 여기 처리가 안 되네

Approach 2
- fixed size array를 사용한다. 길이 300자리 array를 사용한다. 
  - 각 array에는 [timestamp, count] 이 저장되어 있다.
- hit 처리
  - (new timestamp % 300) 의 위치를 업데이트해야한다. 
    - 기존에 timestamp 가 동일하다면 count 만 하나 증가시킨다.
    - 다르다면 count 를 1로 초기화한다.
- get hit 처리
  - buffer를 scan 하면서 timestamp - 299 이상인 것들의 합을 구한다.


```python
class HitCounter:
    def __init__(self):
        self.SEARCH_RANGE = 300  # Get hits during the last 300 seconds [timestamp-299, timestamp]
        self.hit_buffer = [None] * self.SEARCH_RANGE  # [timestamp, current count] 
    
    def hit(self, timestamp: int) -> None:
        idx = timestamp % self.SEARCH_RANGE
        if self.hit_buffer[idx] and self.hit_buffer[idx][0] == timestamp:
            self.hit_buffer[idx][1] += 1
            return
        self.hit_buffer[idx] = [timestamp, 1]
            
                
    def get_hits(self, timestamp):
        count_sum = 0
        for ts, cnt in self.hit_buffer:
            if ts > timestamp - self.SEARCH_RANGE:
                count_sum += cnt
        return count_sum
          
    
    
    


```

</details>




