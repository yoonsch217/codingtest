### Array

```java
int[] nums = new int[n];
Arrays.fill(nums, 0);
```

### Hash

hash set

```java
import java.util.HashSet;
import java.util.Set;


Set<Integer> set = new HashSet<>();
// Adding elements to the set
set.add(1);

for (int item : set) {
    System.out.println(item);
}

for (int i : map.values()) {
    // pass
}

boolean isContain = map.containsValue(target);

set.remove(valueToRemove);
```

hash map

```java
import java.util.HashMap;
import java.util.Map;

Map<String, Integer> map = new HashMap<>();  
// new Map<>(); 는 compilation error. cannot instantiate Interface Map.
// HashMap<String, Integer> map = ... 는 가능하고 동일한 기능이다.
// HashMap<String, int> map 은 compilation error. Generics in Java do not support primitive types; they only support reference types

// Adding key-value pairs to the map
map.put("apple", 10);
int count = map.get("apple");
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

for (String key : map.keySet()) {
    System.out.println("Key: " + key);
}

```


### Queue

```java
import java.util.LinkedList;
import java.util.Queue;


Queue<Integer> queue = new LinkedList<>();
queue.add(2);     // queue에 값 추가. 성공하면 true를 반환하고 실패하면 exception이 발생한다.
queue.offer(3);   // queue에 값 추가. 성공하면 true, 실패하면 false를 반환한다.

queue.peek(); // queue의 첫 번째 값을 얻는다. 비어있다면 null이 반환된다.

queue.poll();       // queue의 첫번째 값 제거하고 그 값을 반환한다. 비어있다면 exception이 반환된다.
queue.remove();     // queue의 첫번째 값 제거하고 그 값을 반환한다. 비어있다면 null이 반환된다.
queue.clear();      // queue 초기화

```


### Stack

Vector 클래스를 상속받기 때문에 thread-safe하다는 특징이 있다.

```java
import java.util.Stack;

Stack<Integer> stackInt = new Stack<>();
stackInt.push(1);
stackInt.add(2);

stackInt.search(2);  // 몇 번 pop을 해야 parameter가 나오는지를 반환한다. 여러 개인 경우 처음으로 나오는 순서를 반환한다. 값이 없는 경우 -1이 반환된다. 맨 뒤에 있는 경우는 1이 반환된다.

stackInt.peek();  // 스택의 마지막 요소를 반환한다. 비어있는 경우는 Exception이 발생한다.
stackInt.pop();  // 마지막 요소를 제거하고 반환한다.
stackInt.clear();

stackInt.isEmpty();  
stackInt.size();
stackInt.contains(1);
```

하지만 vector 클래스의 add 메소드도 사용될 수 있는데 그러면 의도치 않은 동작이 발생할 수 있다.   
예를 들어, stack.push(1); stack.push(2); stack.add(0, 3); 와 같은 코드를 실수로 썼다면 add 함수는 제일 앞에 3이란 원소를 넣을 것이다. 그러면 stack.peek();는 2가 된다.

자바 공식문서에서는 Stack보다 Deque를 사용할 것을 권장한다.


### Deque

양방향 큐 자료구조. queue, stack 둘 다 구현 가능하다. Double-ended Queue   
여러 연산들을 정의한 Deque 인터페이스가 있고 이를 구현한 ArrayDeque, LinkedBlockingDeque, ConcurrentLinkedDeque, LinkedList 등의 클래스가 있다. 

원소 추가
- addFirst(): 앞쪽에 넣는다. 용량을 초과하면 exception이 발생한다.
- offerFirst(): 앞쪽에 넣는다. 성공하면 true, 실패하면 false를 반환한다.
- addLast(): 뒤에 넣는다. 용량을 초과하면 exception이 발생한다.
- offerLast(): 뒤에 넣는다. 성공하면 true, 실패하면 false를 반환한다.
- add(): addLast와 동일하다.
- offer(): offerLast와 동일하다.
- push(): addFirst와 동일하다. 

원소 제거
- removeFirst(): 앞쪽 하나를 제거하고 반환한다. 비어있으면 exception이 발생한다.
- pollFirst(): 앞쪽 하나를 제거하고 반환한다. 비어있으면 null이 반환된다.
- removeLast(): 뒤에서 하나를 제거하고 반환한다. 비어있으면 exception이 발생한다.
- pollLast(): 뒤에서 하나를 제거하고 반환한다. 비어있으면 null이 반환된다.
- remove(): removeFirst와 동일하다.
- poll(): pollFirst와 동일하다.
- pop(): 이것도 앞에 꺼 제거하고 반환한다.

원소 얻기
- getFirst(): 맨앞을 반환한다. 비어있으면 exception이 발생한다.
- peekFirst(): 맨앞을 반환한다. 비어있으면 null이 반환된다.
- getLast(): 맨뒤를 반환한다. 비어있으면 exception이 발생한다.
- peekLast(): 맨뒤를 반환한다. 비어있으면 null이 반환된다.
- peek(): peekFirst와 동일하다.

기타
- removeFirstOccurence(Object o): 앞쪽부터 탐색하여 o와 동일한 첫 원소를 제거한다. 없다면 아무일도 일어나지 않는다.
- removeLastOccurence(Object o): 뒤부터 탐색한다.
- size(): 크기를 반환한다.
- contain(Object o): o가 있는지 반환한다.


push가 앞에 원소를 넣고 pop이 앞에 원소를 빼네. stack 느낌이 아니다.

정리

- offerFirst()
- offerLast()
- pollFirst()
- pollLast()
- peekFirst()
- peekLast()



```java
import java.util.Deque;
import java.util.ArrayDeque;

class HelloWorld {
    public static void main(String[] args) {
        Deque<Integer> deque = new ArrayDeque<>();
        deque.add(2);
        deque.add(3);
        deque.offer(4);
        deque.offerFirst(1);
        deque.offerLast(5);
        
        System.out.println(deque);  // [1, 2, 3, 4, 5]
        System.out.println(deque.peek());  // 1
        System.out.println(deque.peekFirst());  // 2
        System.out.println(deque.getLast());  // 5
        
        System.out.println(deque.remove());  // 1
        System.out.println(deque.poll());  // 2
        System.out.println(deque.pollLast());  // 5
        
        System.out.println(deque.size());  // 2
        
        System.out.println(deque.pollFirst());  // 3
        System.out.println(deque.removeLast());  // 4
        System.out.println(deque.pollFirst());  // null
        System.out.println(deque.size());  // 0
    }
}
```



### Heap

Primary Queue는 우선순위 큐로 우선순위가 가장 낮은 값이 먼저 나오게 되어 있다.    
우선순위가 낮다는 것은 Integer를 넣었을 때, 최소값으로 판단된다.

```java
PriorityQueue<Integer> minHeap = new PriorityQueue<>();

```

최대 힙을 사용하는 방법은 Comparator 값을 세팅해주어 사용할 수 있다.

```java
import java.util.Comparator;

PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(new Comparator<Integer>() {
        @Override
        public int compare(Integer o1, Integer o2) {
            return - Integer.compare(o1,o2);
        }
    });
```




### Graph


### Sort



