아래와 같이 정의하는 게 유연한 코드이다.

```java
MyInterface<T> var = new MyClass<>();
```


### Array

```java
int[] nums = new int[n];
Arrays.fill(nums, 0);
int n = nums.length;  // length로 길이를 구한다.

int[] nums2 = {1, 2, 3};  // initialize 할 땐 {}를 사용한다.

Arrays.sort(nums);  // ArrayList는 안 된다.
System.out.println(Arrays.toString(nums));
```

```java
List<Integer> arr = new ArrayList<Integer>(4); // 이렇게 정의하면 size 4짜리를 만든다. default는 10이다. 4를 넘어도 resize를 한다. 초기 상태에서 size() 호출하면 0이다.
arr.add(10);
arr.add(20);
arr.add(30);
arr.add(40);

int target = arr.get(2);  // 30

Collections.sort(arr);
int n = arr.size();  // size()로 길이를 구한다.

```











### Hash

hash set

```java
import java.util.HashSet;
import java.util.Set;


Set<Integer> set = new HashSet<>();

set.add(1);  // Adding elements to the set

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

map.put("apple", 0);
map.put("apple", 10);  // update
map.put("apple", map.get("apple") + 10);  // update

// update when exists
map.put("apple", map.containsKey("apple") ? map.get("apple") + 10 : 0);  
map.put("banana", map.getOrDefault("banana", 0) + 10);

int count = map.get("apple");
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

for (String key : map.keySet()) {
    System.out.println("Key: " + key);
}


// adjacent list 구할 때
Map<Integer,List<int[]>> map=new HashMap<>();

```


### ArrayList, HashMap 예시


<details><summary>1094. Car Pooling</summary>


```java
class Solution {
    public boolean carPooling(int[][] trips, int capacity) {
        HashMap<Integer, Integer> timeToNums = new HashMap<>();

        int n = trips.length;
        for(int i = 0; i < n; i++) {
            int numPassengers = trips[i][0];
            int fromTime = trips[i][1];
            int toTime = trips[i][2];

            // HashMap update 방법 1
            if (timeToNums.containsKey(fromTime)) {
                int prevNum = timeToNums.get(fromTime);
                timeToNums.put(fromTime, prevNum + numPassengers);
            }
            else {
                timeToNums.put(fromTime, numPassengers);
            }

            // HashMap update 방법 2
            timeToNums.put(toTime, timeToNums.getOrDefault(toTime, 0) - numPassengers);
            }
        
        // keySet을 ArrayList로 받기
        List<Integer> timeArray = new ArrayList<>(timeToNums.keySet());

        // ArrayList 정렬하기
        // 역순 정렬: Collections.sort(timeArray, Collections.reverseOrder()) 
        Collections.sort(timeArray);  
        
        int sumPassengers = 0;
        for(int curTime : timeArray) {
            sumPassengers = sumPassengers + timeToNums.get(curTime);
            if (sumPassengers > capacity) {
                return false;
            }
        }
        return true;
    }
}
```

</details>
















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

pair로 넣기

```java
import javafx.util.Pair; // For JavaFX Pair class

Deque<Pair<Integer, Integer>> deque = new ArrayDeque<>();
deque.offerLast(new Pair<>(value1, value2));
```






### deque 예시


<details><summary>503. Next Greater Element II</summary>

```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];

        Deque<Integer> d_deque = new ArrayDeque<>();

        for (int i = n-1; i >= 0; i--) {
            int cur = nums[i];
            while (!d_deque.isEmpty() && d_deque.peekLast() <= cur) {
                d_deque.pollLast();
            }
            if (d_deque.isEmpty()) {
                res[i] = -1;
            }
            else {
                res[i] = d_deque.peekLast();
            }
            d_deque.offerLast(cur);
        }

        for (int i = n-1; i >= 0; i--) {
            int cur = nums[i];
            while (!d_deque.isEmpty() && d_deque.peekLast() <= cur) {
                d_deque.pollLast();
            }
            if (d_deque.isEmpty()) {
                break;
            }
            else {
                res[i] = d_deque.peekLast();
            }
            d_deque.offerLast(cur);
        }

        return res;
    }
}
```

</details>















### Heap (PriorityQueue)

Primary Queue는 우선순위 큐로 우선순위가 가장 낮은 값이 먼저 나오게 되어 있다.    
우선순위가 낮다는 것은 Integer를 넣었을 때, 최소값으로 판단된다.

메소드는 deque와 유사하다.

- add()
- offer()
- poll()
- remove()
- isEmpty()
- size()

```java
import java.util.PriorityQueue;

PriorityQueue<Integer> minHeap = new PriorityQueue<>();
pq.offer(3);
pq.offer(1);
pq.offer(2);
System.out.println("Top element: " + pq.peek()); // Output: 1

while (!pq.isEmpty()) {
    System.out.println("Removed element: " + pq.poll());
}
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

혹은 다음과 같이 간단히 사용도 가능한 것 같다.

```java
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());

PriorityQueue<Integer> maxHeap = new PriorityQueue<>((o1, o2) -> Integer.compare(o2, o1));
```


큐에 tuple 혹은 list 형태로 넣어야할 때

```java
PriorityQueue<int[]> minHeap = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));
```


PriorityQueue of Pair<K, V> Syntax

```java
// Since Pair<K, V> class was the part of JavaFX and JavaFX was removed from JDK since JDK 11. So Pairs can be used till JDK 10. 
PriorityQueue<Pair<K, V>> = new PriorityQueue<>(initialCapacity, Comparator.comparing(Pair :: getKey));
PriorityQueue<Pair<K, V>> = new PriorityQueue<>(Comparator.comparing(Pair :: getKey));
```





### Heap 예시

<details><summary>378. Kth Smallest Element in a Sorted Matrix</summary>

```java
class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        int n = matrix.length;
        PriorityQueue<int[]> minHeap = new PriorityQueue<>(Comparator.comparingInt(o -> o[0]));

        for (int i = 0; i < n; i++) {
            minHeap.offer(new int[]{matrix[i][0], i, 0});
        }

        int ans = 0;
        for (int i = 0; i < k; i++) {
            int[] top = minHeap.poll();
            int val = top[0], row = top[1], col = top[2];
            ans = val;

            if (col+1 < n) {
                minHeap.offer(new int[]{matrix[row][col+1], row, col+1});
            }
        }

        return ans;
    }
}

```



</details>










### Graph


<details><summary>101. Symmetric Tree</summary>

```java
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return root == null || isSymmetricHelper(root.left, root.right);
    }

    private boolean isSymmetricHelper(TreeNode left, TreeNode right) {
        if (left == null || right == null) {
            return left == right;
        }
        if (left.val != right.val) {
            return false;
        }

        return isSymmetricHelper(left.left, right.right) && isSymmetricHelper(left.right, right.left);
    }
}
```

</details>



