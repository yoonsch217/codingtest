- Nary tree의 특별한 형태이다.   
- 보통 문자열을 저장하는 데 쓰이고 각 trie node는 string을 나타낸다.   
- 각 노드는 여러 자식 노드를 가질 수 있는데 다른 자식 노드로 간다는 건 다른 문자열을 의미한다.   
- 어떤 노드가 의미하는 문자열은, 그 노드까지의 path에 있는 문자에다가 그 노드의 문자를 더한 것이 된다.   
- 루트 노드는 빈 문자열이다.   
- 어떤 노드의 자식들은 동일한 prefix를 갖게 된다. 이 때문에 prefix tree라고도 불린다.   

### How to represent a Trie?

- Array
  - 소문자로만 이루어졌다면 각 노드마다 길이 26짜리의 array를 만들어서 자식 노드를 저장할 수 있다.
  - TrieNode 라는 객체는 길이 26짜리 TrieNode array가 된다. tree라는 단어가 있을 때 root node에서 t 위치에 TrieNode가 존재하게 된다. 없는 곳은 None이 존재해야할 것 같다.   
  - 속도가 빠르지만 메모리를 낭비할 가능성이 있다.

```cpp
struct TrieNode {
    TrieNode* children[N];
    
    // you might need some extra values according to different cases
};
```

- Map
  - array 보다 약간 느릴 수 있지만 메모리를 많이 아낀다.
  - 고정된 길이를 사용하는 게 아니기 때문에 flexible하다.

```cpp
struct TrieNode {
    unordered_map<char, TrieNode*> children;
    
    // you might need some extra values according to different cases
};
```

어떤 node의 map에 어떤 key가 있다는 건, 그 문자를 고를 수 있다는 것이다.
그 문자를 고르면 그에 해당하는 TrieNode로 가게 된다. 이제 그 node의 map에 있는 key가 다음으로 갈 수 있는 후보들이다.   
만약 leaf node라면 해당 노드는 비어있는 map을 갖게 된다.   


### Basic Operations

insertion

```
1. Initialize: cur = root
2. for each char c in target string S:
3.      if cur does not have a child c:
4.          cur.children[c] = new Trie node
5.      cur = cur.children[c]
6. cur is the node which represents the string S
```

search

```
1. Initialize: cur = root
2. for each char c in target string S:
3.   if cur does not have a child c:
4.     search fails
5.   cur = cur.children[c]
6. search successes
```

- 단순히 search 말고 특정 단어가 존재하는지를 물어볼 때도 있다.   
- 예를 들어, binary 라는 문자가 저장되어 있는 trie에서 bin이 있는지를 찾는다면 bin이라는 path까지는 존재한다.   
- 여기서 n이 어떤 단어의 마지막인지에 대한 정보가 있어야한다.   
- isEnd와 같은 flag를 사용할 수 있다.
  - 하나의 TrieNode 를 list로 볼 때, 소문자라면 크기 26의 list를 정의하면 된다. 이 때, 27번째 element를 is_end 를 나타내는 boolean 값으로 넣어주면 된다.
  - 혹은 더 좋은 방식으론 TrieNode 에 children 필드와 별개로 is_end 필드를 보관하는 것이다.

delete
- 삭제는 좀 더 복잡하다.
  - 단어가 Trie에 아예 없는 경우: 아무 작업도 하지 않는다.
  - 단어가 다른 단어의 접두사(Prefix)인 경우: (예: binary가 있는데 bin을 삭제)
    - 이때는 n 노드의 is_end만 False로 바꾸면 끝난다. 하위 노드(a, r, y)는 건드리면 안 된다.
  - 단어가 고유한 경로를 가진 경우: (예: apple을 삭제하는데 다른 a...로 시작하는 단어가 없는 경우)
    - 이때는 리프 노드부터 부모 쪽으로 올라오며 자식이 없는 노드들을 차례로 삭제해야 메모리가 절약된다.

```python
def delete(node, word, depth):
    # 1. 기저 사례: 단어의 끝에 도달했을 때
    if depth == len(word):
        # 단어가 존재한다면 is_end를 해제
        if node.is_end:
            node.is_end = False
        
        # 만약 이 노드에 자식이 하나도 없다면, 부모에게 나를 지워달라고(True) 반환
        return len(node.children) == 0

    char = word[depth]
    if char not in node.children:
        return False # 삭제할 단어가 없음

    # 2. 재귀적으로 자식 노드로 내려감
    should_delete_child = delete(node.children[char], word, depth + 1)

    # 3. 자식 노드를 지워야 한다고 판명되면 삭제 실행
    if should_delete_child:
        del node.children[char]
        
        # 자식을 지운 후, 현재 노드도 지워질 조건이 되는지 확인:
        # "나도 단어의 끝이 아니고, 남은 자식도 없다면" 나도 지워져야 함
        return not node.is_end and len(node.children) == 0

    return False
```







