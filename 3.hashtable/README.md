## 개념

### 구현

Hash Table을 구현하기 위해서는 아래의 과정이 필요하다.
1. key의 hash code를 계산하는데 hash code는 보통 숫자이다. hash code 범위는 유한하기 때문에 서로 다른 두 키가 같은 hash code를 가질 수 있다.
2. hash code를 array의 index에 매핑한다.
3. 각 index에는 key와 value의 linked list가 있다.

데이터를 얻기 위해서는 아래의 과정이 필요하다.
1. key를 통해 hash code를 계산한다.
2. hash code로 index를 구한다.
3. 해당 index의 linked list에서 값을 찾는다.

lookup time complexity는 O(N/k)이다. k는 hash function을 수행했을 때 나올 수 있는 key의 수이다.
collision이 많다면 worst case runtime은 O(N)이 될 수 있다. 하지만 일반적으로 잘 구현되어 collision이 최소로 됐다고 가정한다면 lookup time은 O(1)이다.

또, hash table을 balanced binary search tree로도 구현할 수 있다.   
이렇게 구현하면 lookup time은 O(log N/k)이 된다.    
k개의 key가 있고 각 key마다 linked list가 아닌 binary search tree를 놓는 것이다. bst에서의 lookup time은 height인 log N이다.   
그럼 결국 linked list보다 더 빠른 거 아닌가?   
이렇게 할 때의 장점은 메모리를 적게 쓴다는 점과 key를 iterate하기 편하다는 점도 있다.   

### hash set / hash map

hash table이 더 넓은 개념이다.   
hash function을 이용한 자료 구조를 hash table이라 하고 그걸 구현하는 방법에 따라 hash set과 hash map으로 나눌 수 있다.

- hash set: 중복된 원소가 없도록 한 자료구조이다.
- hash map: key 뿐 아니라 key, value pair로 이루어진 자료구조이다.



## 전략