### 208. Implement Trie (Prefix Tree)

https://leetcode.com/problems/implement-trie-prefix-tree/description/

문제: Trie의 insert, search, startsWith 메소드를 구현하라.

search의 경우는 해당 단어가 이전에 insert 됐어야 true인 것이다.   
따라서 일반 trie 구조로 하려면 is_end 와 같은 bool flag가 있어야한다.   
이 flag가 True라면 그 노드에서 끝난 word가 있다는 것이다.   

나는 좀 더 메모리 써서 했다.

<details>

```py
class Trie:
    def __init__(self):
        self.children = {}
        self.words = set()
        

    def insert(self, word: str) -> None:
        self.words.add(word)
        
        first_c = word[0]
        if first_c not in self.children:
            self.children[first_c] = Trie()
        self.children[first_c].insert_helper(word, 1)
    
    def insert_helper(self, word: str, idx: int) -> None:
        if idx == len(word):
            return
        head = word[idx]
        if head not in self.children:
            self.children[head] = Trie()
        self.children[head].insert_helper(word, idx+1)

    def search(self, word: str) -> bool:
        return word in self.words

    def startsWith(self, prefix: str) -> bool:        
        first_c = prefix[0]
        if first_c not in self.children:
            return False
        return self.children[first_c].startsWith_helper(prefix, 1)
    
    def startsWith_helper(self, prefix: str, idx: int) -> bool:
        if idx == len(prefix):
            return True
        head = prefix[idx]
        if head not in self.children:
            return False
        return self.children[head].startsWith_helper(prefix, idx+1)
```

`word[1:]` 이렇게 slice해서 넘기다가 index로 넘기니까 더 빨라졌다.   

</details>


반면에 Trie는 그냥 커다란 트리로 두고 node 객체를 따로 만드는 방법도 있는데 이게 더 빠르고 좋아보인다.

```py
class TrieNode:
    def __init__(self):
        # Stores children nodes and whether node is the end of a word
        self.children = {}
        self.isEnd = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        # Insert character by character into trie
        for c in word:
            # if character path does not exist, create it
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.isEnd = True
        

    def search(self, word: str) -> bool:
        cur = self.root
        # Search character by character in trie
        for c in word:
            # if character path does not exist, return False
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.isEnd
        

    def startsWith(self, prefix: str) -> bool:
        # Same as search, except there is no isEnd condition at final return
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True
```




