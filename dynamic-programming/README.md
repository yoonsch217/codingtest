

## 문제

### Min Cost Climbing Stairs
https://leetcode.com/problems/min-cost-climbing-stairs

문제: 각 계단마다 cost가 있고 한 칸, 혹은 두 칸을 이동할 수 있다. 최소의 cost로 끝까지 갈 때 cost를 구하라.

어떤 i번째 층에 간다고 했을 때, 답은 최소로 한 칸 밑으로 갈 때와 최소로 두 칸 밑으로 갈 때 중 작은 값이다.    
왜냐하면, 한 칸 아래에 있든, 두 칸 아래에 있든 현재 층으로 올 수 있기 때문이다.   
dp(i)를 i번째 칸에 도착하기 위한 최소의 cost라고 하자.   
그러면 `dp(i) = min(dp(i-2)+cost[i-2], dp(i-1)+cost[i-1])`가 일반식이 된다.   
구해야하는 답은 dp(len(cost))가 된다. 


### Word Break
https://leetcode.com/problems/word-break/

문제: 문자열 s가 있고 word_dict이라는 문자열 리스트가 있다. word_dict에 있는 문자열들로 s를 만들 수 있는지 판단하라.

dp(i): i~end까지의 결과    
dp(0)을 구하면 된다.   
dp(i)에 대해 j를 i+1부터 끝까지 이동시키면서 s[i:j]가 word_dict에 있는지 확인하고 있으면 dp(j)를 반환하면 된다.   

O(N^2) /  O(N)






