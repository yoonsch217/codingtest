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

https://leetcode.com/problems/word-break-ii/description/

문제: Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.

Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]   
Output: ["cats and dog","cat sand dog"]

<details><summary>Approach 1</summary>

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



```python
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

trie + memoization

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        # 1. 반복문을 이용한 효율적인 Trie 구성
        trie = {}
        for word in wordDict:
            node = trie
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['is_end'] = True

        # 2. 메모이제이션을 위한 딕셔너리
        memo = {}

        def dfs(start_idx):
            # 이미 계산된 결과가 있다면 반환
            if start_idx in memo:
                return memo[start_idx]
            
            # 문자열 끝에 도달하면 빈 문자열을 담은 리스트 반환 (조합의 기초)
            if start_idx == len(s):
                return [""]

            res = []
            node = trie
            
            # 현재 인덱스부터 Trie를 따라가며 가능한 단어 탐색
            for i in range(start_idx, len(s)):
                char = s[i]
                if char not in node:
                    break  # Trie에 없는 문자면 더 이상 탐색 불가능
                
                node = node[char]
                
                # 단어를 하나 찾았다면, 그 이후의 문자열에 대해 재귀 호출
                if 'is_end' in node:
                    word = s[start_idx : i + 1]
                    sub_results = dfs(i + 1)
                    
                    for sub in sub_results:
                        if sub == "": # 끝에 도달한 경우
                            res.append(word)
                        else: # 뒤에 이어질 문장이 있는 경우
                            res.append(word + " " + sub)
            
            memo[start_idx] = res
            return res

        return dfs(0)
```

</details>


<details><summary>Approach 2</summary>

내가 제일 먼저 생각난 건 dp였다.

```python
def wordBreak(self, s: str, word_dict: List[str]) -> List[str]:
    """
    dp[i] = list of all possible outputs using s[:i+1]
    dp[i] = for all word in dict, dp[i - len(word) + 1] + " " + word
    """

    dp = []
    for i, c in enumerate(s):
        cur_list = []
        for word in word_dict:
            if len(word) > i + 1:
                continue
            if s[i-len(word)+1:i+1] != word:
                continue
            prev_idx = i - len(word)
            if prev_idx == -1:
                cur_list.append(word)
                continue
            if len(dp[prev_idx]) == 0:
                continue
            for prev_s in dp[prev_idx]:
                cur_list.append(' '.join([prev_s, word]))
        dp.append(cur_list)
    return dp[len(s)-1]
```

</details>




