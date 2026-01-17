# 개념

정렬된 배열에서 중간값과 비교하며 탐색 범위를 절반씩 줄여가는 알고리즘
- 시간 복잡도: O(log N)
- 공간 복잡도: O(1) (반복), O(log N) (재귀)
- 전제 조건: 배열이 정렬되어 있어야 함

### 기본 코드: 일치하는 값 찾기

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid  # 찾음
        elif arr[mid] < target:
            left = mid + 1  # 오른쪽 탐색
        else:
            right = mid - 1  # 왼쪽 탐색
    
    return -1  # 이 때의 right 값은 target 보다 작은 최댓값, left 값은 target 보다 큰 최솟값이 된다.
```

```python
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```


### 만족하는 값 중 가장 앞에 있는 원소 찾기

```python
def find_first(arr, target):
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            result = mid  # 임시로 결과 저장
            right = mid - 1  # 현재 값보다 더 왼쪽에도 있는지 탐색
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result
```

### 조건을 만족하는 값 찾기

binary search할 때    
target 찾는 게 목표인 건 helper(mid) == target 되면 return해버리면 되니까 간단하다.   
어떤 조건을 만족하는 값을 찾는 문제일 때,

```python
while left <= right:
  mid = (left+right) // 2
  
  if condition_a:
    left = mid + 1    
  else:
    right = mid - 1
return left or right

"""
condition_a를 만족하는 건 o, 만족하지 않는 건 x라고 하면
o o o o o x x x 이런 식으로 있을 것이다.
그러면 위 while loop을 나오게 되면 
         
o o o o o x x x
        ^
      right
          ^
         left
가 된다.
즉 right: condition_a를 만족하는 최댓값, left: condition_a를 만족하지 않는 최솟값

x x x x o o 이런 식도 있는데 그러면 not condition_a로 생각하면 된다.
우선 위에 메커니즘을 이해하고 외우면 응용이 편할 것 같다.
"""
```

https://leetcode.com/problems/find-smallest-letter-greater-than-target 의 예시.     
non decreasing order letters에서 target letter 보다 큰 최소의 값을 구하는 문제이다.   
condition은 target보다 작거나 같을 때이다. 그러면 이 condition을 나오게 되면 left pointer가 답이 된다.




### mid 값이 답인지 아닌지 바로 알 수 없는 경우

일반 binary search 에 대해서는 현재 판단하는 mid에 대해서 target과 다르면 mid는 정답이 아니니까 다음 iteration 범위에서 제외시킨다.   
하지만 mid값에 대해서 바로 답인지 아닌지 알 수 없는 경우가 있다.    
정답 포인트가 `mid 오른쪽에 있는지` vs `mid 포함 왼쪽에 있는지` 등의 범위로 나누어지게 된다.

```py
while left < right: 
    if condition(mid): 
        left = mid + 1  # mid는 확실히 답이 아님. 제외하고 오른쪽 탐색 
    else: 
        right = mid  # mid가 답일 수도 있음. 포함해서 왼쪽 탐색
``` 

이런 형식으로 사용해야할 수 있다.   
중간에 return이 없고 iteration을 모두 빠져나와서 left == right 일 때 left or right 가 답이다.

위에 있는 find_first 방식과 서로 대체될 수 있다. 중간 결괏값을 저장하냐 마냐의 차이이다. 

예시: `658. Find K Closest Elements`


### bisect

```python
import bisect

arr = [1, 3, 5, 7, 9, 11, 13]

# 왼쪽 삽입 위치
pos = bisect.bisect_left(arr, 7)
print(pos)  # 3

# 오른쪽 삽입 위치
pos = bisect.bisect_right(arr, 7)
print(pos)  # 4

# 삽입
bisect.insort(arr, 8)
print(arr)  # [1, 3, 5, 7, 8, 9, 11, 13]
```

