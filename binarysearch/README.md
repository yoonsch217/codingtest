binary search할 때    
```
while left <= right:
  mid = (left+right)//2
  do_something
  if a:
    left = mid + 1
  else:
    right = mid - 1
return left or right
```
이렇게 하는데 마지막에 left일지 right일지 헷갈릴 때가 많다.    
그럴 때는 o o o x x x 로 생각을 한다.   
end case는 left == right == mid 일 때다.   
그때의 위치는 i == 2 or i == 3일 것이다.   
if condition의 두 개를 해보면 이후에 left는 항상 3, right는 항상 2가 되는 것을 알 수 있다.   


## 문제

https://leetcode.com/problems/koko-eating-bananas/submissions/

문제: piles 라는 리스트가 있고 각 원소는 바나나의 개수이다. 감시자가 h 시간동안 떠나있을 때, 시간당 k의 속도로 바나나를 먹는다. 한번에 하나의 pile만 먹을 수 있다. 전체 바나나를 다 먹기 위한 최소의 k를 구하라.

풀이    
각 pile 먹는 속도는 math.ceil(pile/k) 이다. 따라서 총 걸리는 시간은 `sum of math.ceil(pile/k) for each pile in piles` 이 된다.    
전체 바나나를 다 먹을 수 있는 속도라면 possible, 못 먹으면 impossible이라고 하자.   
최대의 impossible k 보다 하나 크면 최소의 possible k가 되고, 그게 답이 된다.   
따라서 binary search로 풀 수 있다.    
left와 right를 정해야하는데 최소는 1이 되고 최대는 max(piles)가 된다.   
