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



### Maximal Square

https://leetcode.com/problems/maximal-square/

문제: mxn binary matrix가 0 혹은 1로 채워져있다. 1로만 이루어진 가장 큰 정사각형을 찾아라.

dp(i, j)를 matrix[i][j] 위치에서 왼쪽 위로 만들 수 있는 최대의 정사각형이라 정의한다.    
그러면 `dp(i, j) = min(dp(i-1,j), dp(i,j-1), dpIi-1,j-1))` 가 된다.   
먼저 dp(i-1, j), dp(i, j-1) 을 생각해보고 꼬투리를 dp(i-1, j-1) 로 채운다고 접근하면 될 것 같다.   
이렇게 하면 dp로 풀이는 가능하고, 공간 최적화를 하려면 직전 row의 정보만 보관하면 된다.






