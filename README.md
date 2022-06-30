# codingtest


# 21:12
https://leetcode.com/problems/word-break/
"""
input: s: String, worddict: list
output: boolean: true if s can be made from words in worddict
Can use multiple times.

ex) s = "leetcodecode", wordDict = ["leet","code", "le", "et"] : True

only lower english cases
"""

"""
1) recursion
N: length of string s
best O(N)
worst O(2^N)
"""


def solve(s: String, wordDict: List[String]) -> boolean:
    word_set = set(wordDict)

    @lru_cache(maxSize=None)
    def helper(ss): # => start, end
        if len(ss) == 0:
            return True
        tmp = []
        res = []
        for i in range(len(ss)):
            tmp.append(ss[i])
            if ''.join(tmp) in word_set:
                res.append(helper(ss[i+1:]))
        return any(res)
    
    return helper(s)

def wordBreak(self, s, words):
    ok = [True]
    for i in range(1, len(s)+1):
        ok += any(ok[j] and s[j:i] in words for j in range(i)),
    return ok[-1]

"""
ex) s = "leetcodecode", wordDict = ["leet","code", "le", "et"] 


"leetcode"
["leet","code"]

"""








---



https://leetcode.com/problems/boats-to-save-people/   
https://leetcode.com/problems/koko-eating-bananas/solution/   
https://leetcode.com/problems/maximal-square/   

minimum spanning tree
