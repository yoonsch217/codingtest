### 208. Implement Trie (Prefix Tree)

https://leetcode.com/problems/implement-trie-prefix-tree/description/

문제: Trie의 insert, search, startsWith 메소드를 구현하라.



Trie는 그냥 커다란 트리로 두고 node 객체를 따로 만드는 방법도 있는데 이게 빠르고 좋아보인다.

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









### 140. Word Break II

문제: Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.

Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]   
Output: ["cats and dog","cat sand dog"]


- word dict의 단어들로 Trie를 만든다.
- s를 iterate하면서 Trie를 따라간다. 
- matching하는 단어를 찾았으면 두 분기로 나눈다.
   - 그곳에 space를 넣고 이후 탐색은 다시 Trie root부터 시작
   - space를 넣지 않고 Trie를 계속 내려가면서 탐색
- s의 마지막 시점에 TrieNode에 is_end=True 가 만족해야 올바른 word break가 된다.

포인트가 두 개 있는 것 같다.    
- Trie를 어떻게 light-weight로 구현할 것인가. 
   - multi-depth dict로 구현을 했다. 클래스를 만드는 것 보다 간편하고 가볍다.
- ans array를 어떻게 업데이트할 것인가.
   - list 하나를 계속 사용했다. backtracking인가 이게? DFS로 쭉 갔다가 나올 때는 list에서 pop 하면서 나왔다. 덕분에 메모리도 아끼고 리스트 복사하는 시간도 아꼈다.


<details>

```py
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        trie = {}

        def insert_into_trie(d, w, i):
            c = w[i]
            if c not in d:
                d[c] = {}
            if i == len(w)-1:
                d[c]['is_end'] = True
                return
            insert_into_trie(d[c], w, i+1)

        for word in wordDict:
            insert_into_trie(trie, word, 0)

        ans = []

        def get_answer(s, i, ans, trie, d, tmp_ans):
            if i == len(s):
                return
            c = s[i]
            if c not in d:
                return
            if 'is_end' in d[c]:
                tmp_ans.append(c)
                tmp_ans.append(' ')
                if i == len(s)-1:
                    ans.append(''.join(tmp_ans).strip())
                get_answer(s, i+1, ans, trie, trie, tmp_ans)
                tmp_ans.pop()
                tmp_ans.pop()
            
            tmp_ans.append(c)
            get_answer(s, i+1, ans, trie, d[c], tmp_ans)
            tmp_ans.pop()
        
        get_answer(s, 0, ans, trie, trie, [])
        return ans

```

</details>




