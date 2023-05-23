Nary tree의 특별한 형태이다.   
보통 문자열을 저장하는 데 쓰이고 각 trie node는 string을 나타낸다.   
각 노드는 여러 자식 노드를 가질 수 있는데 다른 자식 노드로 간다는 건 다른 문자열을 의미한다.   
어떤 노드가 의미하는 문자열은, 그 노드까지의 path에 있는 문자에다가 그 노드의 문자를 더한 것이 된다.   

루트 노드는 빈 문자열이다.   
어떤 노드의 자식들은 동일한 prefix를 갖게 된다. 이 때문에 prefix tree라고도 불린다.   

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








