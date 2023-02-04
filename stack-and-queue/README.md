## 개념

### Stack

LIFO   
stack의 대표적인 함수들: pop(), push(item), peek(), isEmpty()   
array처럼 i-th item 접근에 constant time이 걸리는 건 아니다.    
add, remove에 constant time이 걸린다.   

linked list처럼 구현할 수가 있다.

어떤 recursive 에서는 stack이 유용할 수 있다.   
recurse하면서 stack에 데이터를 넣고 backtrack하면서 하나씩 없애는 방식이다.     
또한 recursive 알고리즘을 iterative 알고리즘으로 바꿀 때 사용할 수도 있다.  

Python에서는 list가 stack의 역할을 한다.   

recursion도 stack을 이용한 것이다. call stack.

### Queue

FIFO   
queue의 대표적인 함수들: add(item), remove(), peek(), isEmpty()   

마찬가지로 linked list로 구현할 수 있다.   
stack과 다르게 멤버 변수로 first와 last를 둘 다 알고 있어야 peek, remove는 first를 보고 add는 last 뒤에 추가한다.   
stack은 top만 알고 있으면 된다.   

queue의 경우는 breadth-first search나 cache 구현에 사용될 수 있다.

Python에서 deque를 사용할 수 있다. 

```python
from collections import deque

deq = deque()
deq.appendleft(10)
deq.append(0)
head = deq.popleft()
tail = deq.pop()
```




## 전략

### Monotonic Stack

순서와 관련해서 숫자의 element를 비교할 때 사용될 수 있는 테크닉이다.   
stack에 모든 원소들이 오름차순 혹은 내림차순으로 존재하게 된다.   
iterate하면서 cur이 stack peek보다 큰 게 나오면 cur보다 큰 게 나올 때까지 pop하면서 처리해준다.   
그 다음에 stack에 cur 넣고 다음으로 넘어간다.   
