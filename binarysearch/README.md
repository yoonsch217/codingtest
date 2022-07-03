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

left는 condition을 만족하는 가장 작은 값이다.   
이렇게 하는데 마지막에 left일지 right일지 헷갈릴 때가 많다.    
그럴 때는 o o o x x x 로 생각을 한다.   
end case는 left == right == mid 일 때다.   
그때의 위치는 i == 2 or i == 3일 것이다.   
if condition의 두 개를 해보면 이후에 left는 항상 3, right는 항상 2가 되는 것을 알 수 있다.   


## Problems

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
0~threshold 까지는 reachable 이고 threshold+1~end 는 unreachable이다.   
이 특성을 이용해서 binary search를 사용할 수 있다.    
다만 매번 정렬을 하면 너무 cost가 크기 때문에 한번 정렬해서 height diff마다 position을 붙여놓는다. 그러고는 linear하게 iterate하면서 현재 찍은 기준 position보다 낮은 index를 가진 경우만 벽돌이나 사다리에서 차감한다.
NlogN

