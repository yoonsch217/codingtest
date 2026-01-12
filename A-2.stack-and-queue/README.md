## 개념

### Stack

개념
- LIFO   
- stack의 대표적인 함수들: pop(), push(item), peek(), isEmpty()   
- array처럼 i-th item 접근에 constant time이 걸리는 건 아니다.    
- add, remove에 constant time이 걸린다.
- linked list처럼 구현할 수가 있다.

활용
- 재귀(Recursion) 관리: 재귀 알고리즘의 동작 원리 자체가 Call Stack을 이용하는 것이다.
- Iterative 전환: 재귀 알고리즘을 반복문(Iterative) 형태로 바꿀 때 명시적인 스택을 사용한다.
- Backtracking: 경로를 탐색하며 데이터를 넣고, 되돌아올 때 하나씩 꺼내며 확인하는 작업에 유용하다.

Python에서는 list가 stack의 역할을 한다.   



### Queue

개념

- FIFO   
- queue의 대표적인 함수들: add(item), remove(), peek(), isEmpty()
- 마찬가지로 linked list로 구현할 수 있다.   
- Stack은 top만 알고 있으면 되지만  Queue는 멤버 변수로 first와 last 위치를 둘 다 알고 있어야 한다.   
- peek, remove는 first를 보고 add는 last 뒤에 추가한다.

활용
- breadth-first search나 cache 구현에 사용될 수 있다.

Python에서는 deque를 사용할 수 있다. 

```python
from collections import deque

deq = deque()
deq.appendleft(10)
deq.append(0)
head = deq.popleft()
tail = deq.pop()
```




## 전략

### Monotonic Stack(decreasing)

스택 내부의 원소들을 항상 내림차순(또는 크지 않은 순서)으로 유지하는 특별한 형태의 스택 전략이다.
단순히 데이터를 쌓는 것이 아니라, 새로운 데이터를 넣을 때 나보다 작은 놈들은 다 쫓아내고 들어가는 것이 핵심이다.

동작 원리: 새로운 원소 x를 삽입할 때,
- 비교: 스택의 top에 있는 원소가 x보다 작은지 확인
- 제거: x가 top보다 크다면, 내림차순 유지를 위해 top을 pop
  - (이 과정을 top이 x보다 크거나 같아질 때까지 반복)
- 삽입: 이제 x를 스택에 push

활용
- 이 구조를 사용하는 가장 큰 이유는 "나보다 큰 다음 원소(Next Greater Element)가 언제 나타나는가?"를 찾기 위해서이다.
- 스택에서 pop되는 원소 입장에서는? 방금 들어오려는 x가 나보다 큰 첫 번째 오른쪽 원소이다.
- 스택에 새로 들어온 x 입장에서는? 현재 스택의 top에 있는 원소가 나보다 큰 첫 번째 왼쪽 원소이다.
- increasing monotonic stack 이라면?
  - 만약 increasing monotonic stack 이라면, 점점 증가하다가, 낮은 게 들어오면 다 pop 
  - pop 되는 것 입장에서는 현재 값이 나보다 작은 첫 오른쪽 원소이다. 
  - push 되는 것 입장에서는 top 값이 나보다 작은 첫 왼쪽 원소이다. 
