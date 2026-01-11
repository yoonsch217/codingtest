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











