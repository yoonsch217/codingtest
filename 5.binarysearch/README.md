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
```

https://leetcode.com/problems/find-smallest-letter-greater-than-target 의 예시. target보다 큰 최소의 값을 구하는 문제이다.   
condition은 mid위치의 값이 target보다 작거나 같을 때이다. 그때는 답이 오른쪽에 있으므로 left를 옮겨야한다. 이 loop가 끝나면 left는 condition_a를 만족하지 않는 위치에 있거나 array를 넘어서야한다. 즉, left 위치의 값이 target보다 크면서 가장 왼쪽에 위치한다. 왜냐하면 condition을 만족하면 left는 계속해서 오른쪽으로 이동하기 때문이다.   

마지막에 left일지 right일지 헷갈릴 때가 많다.    
그럴 때는 o o o x x x 혹은 x x x o o o로 생각을 한다.   

~end case는 `left == right == mid` or `left == mid, right = left + 1` 일 때다.   
그때의 위치는 i == 2 or i == 3일 것이다.   
if condition의 두 개를 해보면 이후에 left는 항상 3, right는 항상 2가 되는 것을 알 수 있다.~   
(지금 보니까 무슨 말인지 모르겠네)

https://leetcode.com/problems/koko-eating-bananas 문제의 경우 예시   
loop를 나오면 left = right + 1 의 위치에 있다.   
값이 크면 다 먹을 수 있으므로 x x x o o o 의 형태이다.   
left가 오른쪽으로 옮겨져야할 조건인 if condition은, 그때의 mid보다 더 많이 먹으러 가야하는 상황이므로 다 먹지 못하는 경우일 것이다.    
마지막에 if condition을 거쳤다면 mid는 x일 것이므로 left는 가장 왼쪽의 o가 된다.   
마지막에 else condition을 거쳤다면 mid는 o일 것이므로 right는 가장 오른쪽의 x가 되고 left는 가장 왼쪽의 o가 된다.    


### closest

closest 에 대한 binary search 할 때는 `while left <= right: left = mid + 1 or right = mid - 1` 로 하는 게 복잡한 것 같다.   
`while left < right: if A, left = mid + 1, else right = mid` 이런 식으로 하자. A는 0 ~ mid 가 답이 될 수 없는, 답에서 멀다는 조건이다. 

예시: `658. Find K Closest Elements`

