binary search할 때    
```
while left <= right:
  mid = (left+right) // 2
  do_something
  if a:
    left = mid + 1
  else:
    right = mid - 1
return left or right
```

left는 condition을 만족하는 가장 작은 값이다.   
이렇게 하는데 마지막에 left일지 right일지 헷갈릴 때가 많다.    
그럴 때는 o o o x x x 로 생각을 한다.   
end case는 left == right == mid 일 때다.   
그때의 위치는 i == 2 or i == 3일 것이다.   
if condition의 두 개를 해보면 이후에 left는 항상 3, right는 항상 2가 되는 것을 알 수 있다.   

