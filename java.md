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