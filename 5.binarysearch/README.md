# 개념


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

mid값에 대해서 바로 답인지 아닌지 알 수 없는 경우가 있다.    
정답 포인트가 `mid 오른쪽에 있는지` vs `mid 포함 왼쪽에 있는지` 등의 범위로 나누어지게 되는데 이 때는 left == right 일 때가 답일 수 있다.   
따라서 이 때는 

```py
while left <= right: 
    if condition:
        left = mid + 1
    else: 
        right = mid - 1
``` 

대신

```py
while left < right: 
    if condition: 
        left = mid + 1 
    else: 
        right = mid
``` 

이런 형식으로 사용해야할 수 있다.

예시: `658. Find K Closest Elements`

