### 3. Longest Substring Without Repeating Characters

https://leetcode.com/problems/longest-substring-without-repeating-characters

문제: 문자열이 주어졌을 때 반복되는 글자가 없는 가장 긴 substring의 길이를 반환하라.

sliding window를 사용한다. substring의 left와 right를 정해주는 포인터 두 개를 저장한다.   
그리고 dictionary 하나를 만들어서 key는 나타났던 문자, value는 그 문자의 위치를 저장한다.   
string을 traverse하면서 현재 문자가 dictionary에 있으면서 그 문자의 위치가 cur_idx보다 크거나 같으면 left pointer를 `d[cur] + 1` 로 업데이트한다.   
그리고 `d[cur] = right` 로 업데이트 혹은 추가를 해주고 `right += 1` 로 포인터 위치를 옮긴다.   



### 1094. Car Pooling

https://leetcode.com/problems/car-pooling/

문제: capacity 라는 integer와 passengers라는 리스트가 주어진다. passengers 리스트는 `[승객수, 탑승 시각, 하차 시각]` 로 구성되어 있다. 전체 여정을 하면서 capacity를 넘는 순간이 있다면 false, 그렇지 않다면 true를 반환한다.

어떤 시간 순서열에서 i ~ j 까지 어떤 변화가 있다가 사라져야한다면 map[i] += event, map[j] -= event 식으로 사용한다.   
그러면 그 map을 key로 sort하면 시간 순서대로 정렬이 된다. froms와 tos를 iterate하면서 확인한다.   
`dd = sorted(d.items())` 이렇게 하면 dict가 key로 정렬된 후에 tuple list로 저장이 .   
혹은 시각이 1 ~ 1000까지라는 제약이 있다면 bucket sort를 사용할 수도 있다. 


### 2158. Amount of New Area Painted Each Day

https://leetcode.com/problems/amount-of-new-area-painted-each-day/

문제: 0-indexed 2D integer array paint가 주어진다. paint[i] = [start_i, end_i] 인데 i 시점에 start position과 end position 사이를 칠할 수 있다. paint array를 앞에서부터 iterate하면서 해당 범위 내에 색칠을 하는데 같은 부분을 두 번 이상 색칠하지는 않는다. 각 iteration에서 색칠해아 하는 칸의 수를 구하라.

sweep line이라는 개념이 들어간다.   
https://leetcode.com/problems/amount-of-new-area-painted-each-day/discuss/1740812/Python-Complete-3-solutions-using-different-data-structures   
좀 어렵다.   
먼저 iteratre하면서 각각의 start position과 index를 tuple로 묶어서 리스트에 넣는다. end position과 index도 마찬가지로 넣는다. 각각이 start인지 end인지 boolean 등으로 표시를 해놓는다.    
그 다음에 position을 기준으로 sort를 한다.   
그러면 그 리스트를 앞에서부터 스캔하면 빠른 position부터 나올 것이다.   
그다음에는 전체 길이에 맞는 buffer array를 만든다.   
그러고는 position array를 iterate하면서 start가 나오면 그때부터 buffer의 해당 position에 해당 index를 넣는다. end가 나오면 해당 index에 대해서 그만 넣는다.   
이렇게 하면 나중에 buffer array를 살펴봤을 때 어떤 칸에 1, 5가 있다면 인덱스 1 작업과 인덱스 5 작업에 의해 색칠될 수 있던 공간이라는 뜻이다.   
인덱스 낮은 작업이 우선이므로 인덱스 1의 작업에 의한 색칠 부분으로 인식하면 된다.   




### 904. Fruit Into Baskets

https://leetcode.com/problems/fruit-into-baskets/description/

문제: fruits 라는 리스트가 주어지고 각 index에 있는 값은 그 위치에 있는 과일을 의미한다. 과일이 리스트대로 일렬로 나열되어 있고 사용자는 어느 한 지점부터 오른쪽으로 과일을 주워담는다. 최대 두 종류의 과일까지 담을 수 있고 그걸 넘어서는 순간 담을 수 없고 멈춰야한다. 최대로 많이 담을 수 있는 과일의 수를 구하라.

최대라는 말에 꽂혀서 dp로 생각하려다가 도저히 일반식이 안 나왔다.   
related topic을 보니까 sliding window가 나와서 그 방법으로 풀었다.   

최대 길이가 2인 dictionary를 만들고 key는 fruit, value는 그 fruit이 지금까지 나온 위치 중 가장 오른쪽 위치를 저장한다. left, right 포인터를 두고 right 포인터를 하나씩 오른쪽으로 옮긴다. 그러다가 3번째의 과일이 나오게 되면 left 포인터를 옮겨야한다. 이 때 dictionary를 이용하는데 dictionary에 있는 두 가지 과일 중 더 왼쪽에 있는 과일을 버려야한다. 그 과일의 위치 바로 다음부터가 동일한 과일이 연속으로 나오기 시작한 위치이기 때문에 거기에 left 포인터를 놓고 현재 right의 과일을 dictionary에 추가한다. 그러면 두 종류의 과일이 유지된다.

<details>
  
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
            fruit_to_drop = min(d, key=d.get)
            left = d[fruit_to_drop] + 1  # This is the position where the other fruit starts to appear consecutively
            del d[fruit_to_drop]
        d[cur] = right
        res = max(res, right - left + 1)
        right += 1

    return res
```

</details>

