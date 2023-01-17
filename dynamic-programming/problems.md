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

문제: 문자열 s가 있고 word_dict이라는 문자열 리스트가 있다. word_dict에 있는 문자열들로 s를 만들 수 있는지 판단하라. leetcode, ['leet', 'code']

dp(i): 문자열s의 i~end까지 substring에 대한 결과(0~i로 해도 된다.)    
dp(0)을 구하면 된다.   
dp(i)에 대해 j를 i+1부터 끝까지 이동시키면서 s[i:j]가 word_dict에 있는지 확인하고 있으면 dp(j)를 반환하면 된다.   
근데 substring 자르고 dp recursive하게 부를 때 index 설정하는 게 헷갈린다.   
```
dp(i): [0~i]까지의 substring에 대한 결과
dp(i) = dp(i-j) and exists(s[i-j+1 : i+1]) for j in range(1, i+2), dp(음수) = True
j ranger가 헷갈렸는데 dp(i-1) and exists(s[i:i+1]) 부터 dp(-1) and exists(s[0:i+1]) 가 나오도록 설정해야한다.
```

O(N^2) /  O(N)



### Maximal Square

https://leetcode.com/problems/maximal-square/

문제: mxn binary matrix가 0 혹은 1로 채워져있다. 1로만 이루어진 가장 큰 정사각형을 찾아라.

dp(i, j)를 matrix[i][j] 위치에서 왼쪽 위로 만들 수 있는 최대의 정사각형이라 정의한다.    
그러면 `dp(i, j) = min(dp(i-1,j), dp(i,j-1), dp(i-1,j-1))` 가 된다.   
먼저 dp(i-1, j), dp(i, j-1) 을 생각해보고 꼬투리를 dp(i-1, j-1) 로 채운다고 접근하면 될 것 같다.   
이렇게 하면 dp로 풀이는 가능하고, 공간 최적화를 하려면 직전 row의 정보만 보관하면 된다.


### 152. Maximum Product Subarray

https://leetcode.com/problems/maximum-product-subarray/

문제: 정수로 이루어진 리스트가 있을 때 nonempty contiguous subarray의 product가 최대인 값을 구하라. positive, negative, zero 모두 가능하다.

왼쪽에서 오른쪽으로 포인터를 이동시키면서 그 포인터가 끝이라고 했을 때, 시작은 그 포인터의 왼쪽 중 어딘가가 될 수 있다. 이렇게 포인터를 one pass로 이동시키면 모든 contiguous subarray에 대해 확인할 수 있다.    
그 포인터 i가 오른쪽 끝일 때 가능한 contiguous subarray들 중 max와 min 값을 저장한다.   
`dp(i) returns (that_max, that_min)`   
그러면 i+1에 대해서 dp(i+1) 의 max는 `max(nums[i], dp(i)[0] * nums[i], dp(i)[1] * nums[i])` 가 된다.   
nums[i]가 양수일지 음수일지 모르기 때문에 이전까지의 min, max 둘 다 신경을 써야한다. nums[i]가 양수인지 음수인지에 따라 분기를 나누면 더 빨라질 것 같긴 하다.    
bottom up으로 하는 게 효율적이다.   


### 279. Perfect Squares

https://leetcode.com/problems/perfect-squares/

문제: 어떤 int n이 주어졌을 때 perfect square로만 합쳐서 n을 만들도록 할 때의 perfect square number의 최소의 개수를 구하라. perfect square는 정수의 제곱이다.

my approach:   
dp(i)를 최소 개수라고 할 때, `dp(i) = min(dp(j) + dp(i-j)) where 1 <= j < i/2, or 1 if i is a perfect square` 이다.   
a + b = c라고 할 때 a를 이루는 최소 수가 dp(a)이고 b를 이루는 최소 개수가 dp(b)니까 그 두 개 합한 게 되기 때문이다.   
그런데 이렇게 하면 TLE가 난다.    

`dp(i) = min(dp(i-k)+1) for k in perfect square numbers below i` 로 할 수도 있는데 그래도 TLE가 난다.

근데 dp 표현식은 저게 맞다. 내가 각 iteration마다 i가 perfect square 인지 체크하고 맞으면 1로 한 뒤 continue하는 코드를 넣었는데 이게 오히려 시간을 느리게 했나보다. 이 부분을 제외하니까 시간 내로 들어온다.   
perfect square인 경우가 극히 드물텐데 그걸 위해 연산을 했으니 비효율적이었나보다.   
Time Complexity: O(n sqrt(n))

greedy 방식을 사용할 수도 있다. `result = dp(n, k) for n in [i, .. n]` 이고 dp(n, k)는 k 개의 perfect square로 n을 만들 수 있으면 true를 반환하고 그게 그때의 최적의 답이다.   
`dp(n, k) = dp(n-squarenum, k-1) + 1`      
이걸 증명하는 건 contradiction을 이용할 수 있다. dp(n, i)가 있고 그 뒤에 dp(n, j)가 나왔고 dp(n, j)가 더 작은 수라고 하자. dp(n, j)의 답은 j인데 이는 i보다 작아야한다. 그런데 먼저 수행된 i가 더 작아야하므로 모순이다.   
Time Complexity: O(n^(h/2)) where h is the maximal number of recursion that could happen   
n-ary tree로 생각할 수 있다. 어떤 parent node의 숫자를 기준으로, 그 숫자보다 작은 square number를 뺀 node들을 child node로 갖는다.   

greedy 방식을 n-ary tree로 생각할 때, 각 레벨을 BFS로 탐색하는 것으로 볼 수도 있다.    
레벨이 곧 사용된 perfect square 숫자의 개수이기 때문이다.   


### 70. Climbing Stairs

https://leetcode.com/problems/climbing-stairs/

문제: n 개의 step을 올라가야하고 한 번에 한 개나 두 개의 계단을 오를 수 있다. 총 몇 개의 다른 방법으로 올라갈 수 있는지 구하라.

dp(i)를 i개 올라가는 distinct way의 수라고 하자.   
그러면 dp(i) = dp(i-2) + dp(i-1)이 된다.   
어떤 계단에 가기 위해서는 한 계단 아래에서 한 계단 올라오든가 두 계단 아래에서 두 계단 올라와야 하기 때문이다.   


### 276. Paint Fence

https://leetcode.com/problems/paint-fence

문제: n 개의 기둥과 k 개의 색깔이 있다. 연속해서 세 개의 기둥을 동일한 색으로 칠하지 않으면서 모든 기둥을 칠할 수 있는 방법의 수를 구하라.

dp(i)를 i 개를 칠하는 방법의 수라고 하자.   
i번째를 칠하는 방법의 수는 "i-1번째와 다른 색으로 칠하는 방법의 수"와 "i-1번째와 같은 색으로 칠하는 방법의 수"의 합이다.
i-1번째와 다른 색으로 칠하는 방법의 수는, dp(i-1) * k-1이 된다.   
i-1번째와 같은 색으로 칠하는 방법의 수는, i-2와 i-1이 서로 다른 색으로 칠하는 방법의 수와 같다. 따라서 dp(i-2) * (k-1)이 된다.
따라서 dp(i) = (k-1) * (dp(i-1) + dp(i-2))이다.   


---




https://leetcode.com/problems/delete-and-earn/solution/ 740
숫자 리스트가 있고 거기서 x를 뽑으면 그 값을 얻는데 대신 그 x 하나를 지워야하고 x+1전부와 x-1 전부를 지워야한다. 최대>로 얻을 수 있는 sum은?
dp 일반식을 세우고 싶었는데 dp(i)가 뭘 의미해야할지 몰랐었다. i를 뽑는 개수라고 하기는 i, i+1의 관계가 안 만들어지고.
근데 i를 1~i 까지의 답이라고 생각하면 나온다. house robber의 살짝 변형. 한칸 두칸의 살짝 변형.

https://leetcode.com/problems/longest-increasing-subsequence/solution/
dp 일반식 찾기 어려우면 직접 다 예시를 써보면서 해본다. 좀 긴 것 같아도 겁먹지말고 다 써본다. 다 iterate해야하는 거일 수 있다.


https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/



### 322. Coin Change

https://leetcode.com/problems/coin-change/
