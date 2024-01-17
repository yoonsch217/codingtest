README 개념에 연결된 문제

### 53. Maximum Subarray

https://leetcode.com/problems/maximum-subarray

문제: integer array nums가 주어졌을 때, subarray의 sum이 최대인 sum을 구하라.


가장 기본적인 Kadane's algorithm이다.

<details>

```py
def maxSubArray(self, nums: List[int]) -> int:
    best, buffer = -math.inf, 0

    for num in nums:
        buffer = max(buffer+num, num)
        best = max(best, buffer)
    return best
```

</details>





### 918. Maximum Sum Circular Subarray


https://leetcode.com/problems/maximum-sum-circular-subarray

문제: circular integer array nums가 있다. 동일한 위치의 element가 두 번 이상 나오지 않도록 했을 때의 maximum possible sum of a non-empty subarray를 구하라. 


두 가지 case로 나눌 수가 있다. 아래 둘 중 큰 값이 답이다.
- array 안에 포함되는 subarray 중 답이 있는 경우
- for i > j, nums[i:] 와 nums[:j+1] 을 연결한 array 중 답이 있는 경우 

두 번째는 또 여러 방식으로 풀 수 있다.    
- `total sum - subarray의 합이 최소인 값`을 구하면 두 번째 case 중 가장 큰 sum을 구할 수 있다. subarray가 전체 array가 되어 버리면 empty array가 되므로 안 된다.
- right_max[i]를 nums[i:] 중 subarray의 sum이 가장 큰 값이라고 하자. `best = max(best, prefix_sum[i] + right_max[i+1]) for i in range(n-1)`가 된다.


<details>

```py
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        max_best, min_best = -math.inf, math.inf
        max_buffer = 0
        min_buffer_left, min_buffer_right = 0, 0  # 각각 양 끝 중 하나가 포함되지 않은 subarray이다. 전체 array가 되면 답이 empty array가 되기 때문이다.

        # Kadane's algorithm으로 각각의 max or min을 구한다.
        for i, num in enumerate(nums):
            total_sum += num
            max_buffer = max(max_buffer+num, num)
            max_best = max(max_best, max_buffer)
            if i != n-1:
                min_buffer_left = min(min_buffer_left+num, num)
                min_best = min(min_best, min_buffer_left)
            if i != 0:
                min_buffer_right = min(min_buffer_right+num, num)
                min_best = min(min_best, min_buffer_right)
        
        return max(max_best, total_sum - min_best)
```

O(N) time, O(1) space


```python
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Kadane's Algorithm for a single array answer
        single_answer = -math.inf
        cur = 0
        for num in nums:
            cur = max(num, cur+num)
            single_answer = max(single_answer, cur)
        
        # Circular array answer
        right_max = [-math.inf] * n
        right_max[n-1] = nums[n-1]
        postfix = nums[n-1]
        for i in range(n-2, -1, -1):
            num = nums[i]
            postfix += num
            right_max[i] = max(right_max[i+1], postfix)
        
        prefix = 0
        circular_answer = -math.inf
        for i in range(n-1):
            num = nums[i]
            prefix += num
            circular_answer = max(circular_answer, prefix + right_max[i+1])
        
        return max(single_answer, circular_answer)
```

O(N) time, O(N) space


</details>


잘 생각해보면 두 번째 풀이의 코드를 좀 더 간단히 할 수 있다.


- 만약 minSum이 arraySum과 같다면? 전체 array가 min sum subarray라면 그걸 뺀 subarray는 invalid하다.
- 근데 arraySum == minSum 이라면 arraySum - minSum은 0이 된다. 
  - `arraySum == minSum == 양수`라면, array에 최소 하나의 양수가 있다는 건데 array에 그 양수 하나 밖에 없어야한다. 
  array에 음수가 있다면 그 음수만 골라도 min sum이 음수가 되기 때문에 min sum이 양수가 될 수가 없다. 
  array에 다른 양수가 있다면 덜 합치는 게 minSum이 돼야한다. 양수가 하나만 있다면 그건 normal answer에서 다뤄지니까 linked answer는 무시할 수 있다. 
  - `arraySum == minSum == 음수`라면, minSum이 동일하려면 그 array에는 양수가 없어야한다. 
  양수가 있다면 minSum은 그 양수를 포함하지 않을테고 그러면 arraySum != minSum이 되기 때문이다. 
  다 음수라면 그 값은 normal answer보다 클 수가 없다. normal answer은 그 중에서 가장 작은 값 하나만 골랐을 것이기 때문이다. normal answer이 음수가 되고 linked answer이 0이 되기 때문에 단순히 max(normal answer, linked answer) 하면 안 된다. 따라서 이 case에 대한 예외 처리를 해줘야한다.

<details>

```py
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        max_sum = -math.inf
        min_sum = math.inf

        total_sum = 0
        max_cur = min_cur = 0
        for num in nums:
            max_cur = max(max_cur + num, num)
            min_cur = min(min_cur + num, num)
            total_sum += num

            max_sum = max(max_sum, max_cur)
            min_sum = min(min_sum, min_cur)
        
        if min_sum == total_sum:
            return max_sum
        return max(max_sum, total_sum - min_sum)
```

</details>




# 309. Best Time to Buy and Sell Stock with Cooldown

https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/

문제: prices 라는 리스트가 주어진다. 주식을 사거나 팔 수 있는데 팔고나면 하루 동안 cool down이 필요해서 아무것도 못 한다. 주식을 동시에 두 개 이상 갖고 있을 순 없다. 최대로 만들 수 있는 수익을 구하라. 팔고난 뒤가 아니라도 cool down이 가능하다.

state가 복잡할 때는 각각을 나누고 서로의 상관관계를 구하라.
결국에는 얼마나 복잡하든 특정 상태의 i 시점에 대해 과거와의 점화식을 구하는 문제인 것이다.

There can exist three states:   
- Not having any stock   
- Having a stock   
- Just after selling a stock   

no_stock can be turned into: no_stock or have_stock   
have_stock can be turned into: have_stock or after_sell   
after_sell can be turned into: no_stock   

```
s[i]: Maximum profit for the state at the time i. When buying a stock, the profit is decreased by the amount of the price
no_stock[i] = max(no_stock[i-1], after_sell[i-1])
have_stock[i] = max(have_stock[i-1], no_stock[i-1] - prices[i])
after_sell[i] = have_stock[i-1] + prices[i]
```

<details>

```py
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        no_stock = [0] * n
        have_stock = [0] * n
        after_sell = [0] * n

        have_stock[0] = -prices[0]

        for i in range(1, n):
            no_stock[i] = max(no_stock[i-1], after_sell[i-1])
            have_stock[i] = max(have_stock[i-1], no_stock[i-1] - prices[i])
            after_sell[i] = have_stock[i-1] + prices[i]
        
        return max(no_stock[n-1], max(have_stock[n-1], after_sell[n-1]))
```

</details>














---

REAME에 없는 문제


### 70. Climbing Stairs

https://leetcode.com/problems/climbing-stairs/

문제: n 개의 step을 올라가야하고 한 번에 한 개나 두 개의 계단을 오를 수 있다. 총 몇 개의 다른 방법으로 올라갈 수 있는지 구하라.

dp(i)를 i개 올라가는 distinct way의 수라고 하자.   
그러면 dp(i) = dp(i-2) + dp(i-1)이 된다.   
어떤 계단에 가기 위해서는 한 계단 아래에서 한 계단 올라오든가 두 계단 아래에서 두 계단 올라와야 하기 때문이다.   


### Min Cost Climbing Stairs
https://leetcode.com/problems/min-cost-climbing-stairs

문제: 각 계단마다 cost가 있고 한 칸, 혹은 두 칸을 이동할 수 있다. 최소의 cost로 끝까지 갈 때 cost를 구하라.

일반식을 여러 종류로 둘 수 있다.   

dp(i)를 i-th step을 밟기까지의 최소 cost라고 하면 `dp(i+2) = min(dp(i) + cost[i+2], dp(i+1) + cost[i+2])` 가 되고 답은 `min(dp(n-1), dp(n-2))`가 된다.

<details>

```py
def minCostClimbingStairs(self, cost: List[int]) -> int:
    n = len(cost)
    memo = [0] * n
    memo[0], memo[1] = cost[0], cost[1]

    for i in range(2, n):
        memo[i] = min(memo[i-1], memo[i-2]) + cost[i]
    
    return min(memo[n-1], memo[n-2])
```

</details>

dp(i)를 i-th step의 위치까지 올라갈 수 있는 상태가 되는 데까지 들어가는 최소 cost라고 하면 `dp(i) = min(dp(i-2)+cost[i-2], dp(i-1)+cost[i-1])` 가 되고 답은 `dp(n)`이 된다.    
이게 더 깔끔한 거 같기도 하고.   

복잡도는 O(N) / O(N) 일 것이다.





### Word Break
https://leetcode.com/problems/word-break/

문제: 문자열 s가 있고 word_dict이라는 문자열 리스트가 있다. word_dict에 있는 문자열들로 s를 만들 수 있는지 판단하라. leetcode, ['leet', 'code']

적당히 잘 쪼개는 게 중요하다.

- dp(i): index i 까지의 substring이 word_dict 로 구성이 가능하면 True, 아니면 False    
- dp(i) is True when: `s[0:i+1] in word_dict` or `s[j:i+1] in word_dict and dp(j-1) for any j in range(1, i)`   


<details>

```py
def wordBreak(self, s: str, wordDict: List[str]) -> bool:
    wd_set = set()
    for wd in wordDict:
        wd_set.add(wd)
    
    n = len(s)
    memo = [False] * (n+1)  # For memo[-1] = False
    memo[-1] = True

    for i in range(n):
        for j in range(0, i+1):
            if memo[j-1] and s[j:i+1] in wd_set:
                memo[i] = True
                break
    return memo[n-1]
```

</details>


O(N^2) /  O(N)







### 152. Maximum Product Subarray

https://leetcode.com/problems/maximum-product-subarray/

문제: 정수로 이루어진 리스트가 있을 때 nonempty contiguous subarray의 product가 최대인 값을 구하라. positive, negative, zero 모두 가능하다.

왼쪽에서 오른쪽으로 포인터를 이동시키면서 그 포인터가 끝이라고 했을 때, 시작은 그 포인터의 왼쪽 중 어딘가가 될 수 있다. 이렇게 포인터를 one pass로 이동시키면 모든 contiguous subarray에 대해 확인할 수 있다.    
그 포인터 i가 오른쪽 끝일 때 가능한 contiguous subarray들 중 max와 min 값을 저장한다.   
`dp(i) returns (that_max, that_min)`   
그러면 i+1에 대해서 dp(i+1) 의 max는 `max(nums[i], dp(i)[0] * nums[i], dp(i)[1] * nums[i])` 가 된다.   
nums[i]가 양수일지 음수일지 모르기 때문에 이전까지의 min, max 둘 다 신경을 써야한다.    
bottom up으로 하는 게 효율적이다.

```
max_dp(i): i index가 오른쪽 끝인 subarray 중 product max
min_dp(i): i index가 오른쪽 끝인 subarray 중 product min
max_dp(i) = max(max_dp(i) * num, min_dp(i) * num, num)
min_dp(i) = min(min(i) * num, min_dp(i) * num, num)

```

<details>

```py
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        max_dp = [0] * n
        min_dp = [0] * n

        for i, num in enumerate(nums):
            if i == 0:
                max_dp[i] = num
                min_dp[i] = num
                continue
            max_dp[i] = max(max(max_dp[i-1] * num, min_dp[i-1] * num), num)
            min_dp[i] = min(min(max_dp[i-1] * num, min_dp[i-1] * num), num)
        
        return max(max_dp)
```

어차피 직전 값만 사용하니까 list 대신 prev_max, prev_min를 사용할 수도 있다.   
근데 이 때는 prev_max를 업데이트할 때 new_max 라는 값을 만들어서 업데이트해준 뒤 마지막에 바꿔야한다.   
prev_max = ...
prev_min = ...
이런 식으로 하면 prev_min 계산할 때 업데이트 된 prev_max, 즉 new_max를 사용해서 업데이트를 할 위험이 있다.

</details>








### 279. Perfect Squares

https://leetcode.com/problems/perfect-squares/

문제: 어떤 int n이 주어졌을 때 perfect square로만 합쳐서 n을 만들도록 할 때의 perfect square number의 최소의 개수를 구하라. perfect square는 정수의 제곱이다.


dp(i)를 최소 개수라고 할 때, 아래 두 가지로 정의해볼 수 있다.


- `dp(i) = min(dp(j) + dp(i-j)) where 1 <= j < i/2, or 1 if i is a perfect square`
- `dp(i) = min(dp(i-k)+1) for k in perfect square numbers below i` 이게 더 효율적이다.

a + b = c라고 할 때 a를 이루는 최소 수가 dp(a)이고 b를 이루는 최소 개수가 dp(b)니까 dp(c) = dp(a) + dp(b)가 된다.   



<details>

```py
    def numSquares(self, n: int) -> int:
        dp = [float(inf)] * (n+1)

        def helper(num):
            if dp[num] is not float(inf):
                return dp[num]
            if sqrt(num) == int(sqrt(num)):
                dp[num] = 1
                return dp[num]
            """
            # approach 1
            for i in range(1, num//2 + 1):
                dp[num] = min(dp[num], helper(num-i) + helper(i))
            """
            # approach 2
            for i in range(1, num):
                if sqrt(i) == int(sqrt(i)):
                    dp[num] = min(dp[num], 1 + helper(num-i))
            return dp[num]

        res = helper(n)
        return res
```


둘 다 TLE가 난다. 파이썬의 한계인 것 같다.

</details>



BFS 로 풀 수 있다. 이걸 Greedy라고 보기도 하는 것 같다.

root가 target이라고 할 때 tree 구조로 내려오는 걸 생각해본다. 각 child node로 내려올 때마다 square만큼 빼고 남은 값이 된다. 즉, tree의 한 level을 내려올 때마다 square 하나를 사용한 것이다.   
각 node에서 사용할 수 있는 square가 여러 종류가 있다. 각각에 대해 child node를 만들면서 내려오다가 child node의 값이 square 중 하나라면 그 때의 level의 답이 된다.    

- n 이하의 제곱수들을 구해서 저장한다. n을 구성하는 square 후보들이다.
- BFS의 한 level을 리스트로 정의 한다.
- 처음에는 root부터 시작이니까 [n] 가 초깃값이다.
- 현재 level의 tree node마다 돌면서 제곱수라면 그때의 level을 반환한다. root에서 그 tree node까지의 edge 수는 level이고, level 만큼 square를 사용한 것이다.
- 제곱수가 아니라면 다음 child node로 현재 보는 값에서 제곱수를 뺀 값을 넣어준다.
- 그 level의 작업이 끝나면 다음 level에 대해 작업해준다.


<details>

```py
    def numSquares(self, n: int) -> int:
        if n < 2:
            return n
        usable_squares = []
        i = 1
        while i*i <= n:
            usable_squares.append(i*i)
            i += 1
        cnt = 0
        targets = {n}
        while targets:
            cnt += 1
            next_targets = set()
            for target in targets:
                for square in usable_squares:
                    if target == square:
                        return cnt
                    if square > target:  # 처음에는 x 값이 usable_squares 최솟값인 1보다 클 것이다. 그러다가 x가 y보다 작아진다면 이후 y보다도 다 작을 것이므로 더 볼 필요가 없다.
                        break
                    next_targets.add(target - square)
            targets = next_targets

        return cnt
```

</details>


더 빠른 답   
greedy 방식을 사용할 수도 있다. 근데 아래 방식은 dp 아닌가?   

```
dp(target, k): target을 k개의 square로 만들 수 있으면 True
dp(target, k) = dp(target-num, k-1) for num in square_nums
이렇게 하면 정답은 dp(target, k)를 만족하는 최소의 k 값이다.
```

 `result = dp(n, k) for n in [i, .. n]` 이고 dp(n, k)는 k 개의 perfect square로 n을 만들 수 있으면 true를 반환하고 그게 그때의 최적의 답이다.   
`dp(n, k) = dp(n-squarenum, k-1) + 1`      
이걸 증명하는 건 contradiction을 이용할 수 있다. dp(n, i)가 있고 그 뒤에 dp(n, j)가 나왔고 dp(n, j)가 더 작은 수라고 하자. dp(n, j)의 답은 j인데 이는 i보다 작아야한다. 그런데 먼저 수행된 i가 더 작아야하므로 모순이다.   
Time Complexity: O(n^(h/2)) where h is the maximal number of recursion that could happen   

<details>

```python
    def numSquares(self, n: int) -> int:
        square_nums = [i**2 for i in range(1, int(sqrt(n))+1)]
        
        @lru_cache(maxsize=None)
        def is_divided(target, k):
            if k == 1:
                return target in square_nums
            for num in square_nums:
                if is_divided(target-num, k-1):
                    return True
            return False
        
        for i in range(1, n+1):
            if is_divided(n, i):
                return i
```

n-ary tree로 생각할 수 있다. 어떤 parent node의 숫자를 기준으로, 그 숫자보다 작은 square number를 뺀 node들을 child node로 갖는다.   

greedy 방식을 n-ary tree로 생각할 때, 각 레벨을 BFS로 탐색하는 것으로 볼 수도 있다.    
레벨이 곧 사용된 perfect square 숫자의 개수이기 때문이다.   

</details>






### 276. Paint Fence

https://leetcode.com/problems/paint-fence

문제: n 개의 기둥과 k 개의 색깔이 있다. 연속해서 세 개의 기둥을 동일한 색으로 칠하지 않으면서 모든 기둥을 칠할 수 있는 방법의 수를 구하라.

일반식을 생각하기 위해서는 케이스를 잘 쪼개야한다.   
dp(i)를 i 개를 칠하는 방법의 수라고 하자.   
i번째를 칠하는 방법의 수는 "i-1번째와 다른 색으로 칠하는 방법의 수"와 "i-1번째와 같은 색으로 칠하는 방법의 수"의 합이다.   
i-1번째와 다른 색으로 칠하는 방법의 수는, dp(i-1) * k-1이 된다.   
i-1번째와 같은 색으로 칠하는 방법의 수는, i와 i-1이 같기 때문에 i-2는 달라야한다. 따라서 i-2와 i-1을 서로 다른 색으로 칠하는 방법의 수와 같다. 따라서 dp(i-2) * (k-1)이 된다.    
따라서 dp(i) = (k-1) * (dp(i-1) + dp(i-2))이다.   






### 518. Coin Change II

https://leetcode.com/problems/coin-change-ii

문제: coins 라는 리스트는 사용할 수 있는 coin 종류가 있고 amount라는 int가 있다. coins에 있는 coin으로 amount를 만드는 조합의 수를 구하라.


dp(i)를 i 금액을 만들기 위한 방법 수라고 하자.    
처음에 `dp(i) = sum of dp(i-coin) for coin in coins` 라고 생각했는데 이렇게 하면 동일한 조합도 순서가 다르면 다른 way로 처리를 한다.    

knapsack problem이라고 한다.   
일반식 구할 때 적절히 나누자.   

내 풀이    
dp(i, j)를 number of combinations to make up i with using coins[:j+1] 라고 하자.   
dp(i, j)는 `coin[j]를 하나도 안 쓰고 만드는 법` + `coin[j]를 하나라도 쓰고 만드는 법` 으로 나눌 수 있다.   
coin[j]를 하나도 안 쓰고 i를 만드는 법은 dp(i, j-1)이 된다.    
coin[j]를 하나라도 쓰고 만드는 법은 `dp(i-coin[j], j-1) + dp(i-2*coin[j], j-1), ...`이다.   

<details>

```py
    def change(self, amount: int, coins: List[int]) -> int:
        @lru_cache(maxsize=None)
        def helper(i, j):  # Number of ways to make up i with coins[:j+1]
            if i == 0:
                return 1
            if i < 0 or j < 0:
                return 0
            res = helper(i, j-1)

            cnt = 1  # number of coins[j] uses
            while i - coins[j] * cnt >= 0:
                res += helper(i - coins[j] * cnt, j-1)
                cnt += 1
            return res
        
        return helper(amount, len(coins)-1)
```

accept은 되는데 너무 느리다. amount를 N, len(coins)를 M이라고 할 때 시간은 `O(MxNxN)`? recursion은 `MxN` 번 있고 recursion 안에서 iteration이 N번 있는 거 아닌가.   

</details>


Solution    
dp(i, j)를 두 개로 나눈다.   
`coins[j]를 쓰지 않고 make up 하는 방법` + `coins[j]를 쓰고 make up 하는 방법`   
전자는 dp(i, j-1)이 된다.    
후자는 `i-coin[j]` 까지 만들면 거기서 coin[j]만 추가하면 된다. `i-coin[j]`를 만들 땐 coin[j]를 써도 되니까 dp(i-coin[j], j) 이다.    


<details>

```py
    def change(self, amount: int, coins: List[int]) -> int:
        @lru_cache(maxsize=None)
        def helper(i, j):  # Number of ways to make up i with coins[:j+1]
            if i == 0:
                return 1
            if i < 0 or j < 0:
                return 0
            res = helper(i, j-1) + helper(i-coins[j], j)
            return res
        return helper(amount, len(coins)-1)
```

</details>




Optimized Solution     
처음에 dp array는 모두 0으로 초기화한다.   
어떤 특정 coin a로 갈 수 있는 위치를 미리 다 체크해놓고 이 coin은 다시 쓰지 않는다.    
위치를 이동할 때 원래 있던 곳에서 a만큼 이동을 할텐데 `dp(i) += dp(i-a)`가 된다.    
기존의 dp(i)는 coin a 없이 만들어진 값이기 때문에 dp(i-a)에서 coin a를 써서 i로 오는 방법은 기존의 dp(i)를 만들었던 값과 중복이 없다는 것이 보장된다.    
이 작업을 모든 coin에 대해 다 해준다.   


<details>

```python
def change(self, amount: int, coins: List[int]) -> int:
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for x in range(coin, amount + 1):
            """
            dp[x - coin]: 지금 dp(x-1) 까지는 답이 구해진 상태이다. climbing stairs 처럼 생각을 하면 된다.
            현재 coin을 더 사용할 수 있다면 dp(x)는 기존의 dp(x)에다가 dp(x-coin)을 더한 게 된다.
            climbing stairs 같은 경우는 permutation이지만 지금은 combination이기 때문에 coin을 순서대로 사용해야한다.
            """
            dp[x] += dp[x - coin]  
    return dp[amount]
```

Time: O(len(coins) * amount), Space: O(amount)

</details>








### 91. Decode Ways

https://leetcode.com/problems/decode-ways/description/

문제: 숫자 1부터 26은 각각 A부터 Z까지 매핑될 수 있다. 숫자로 이루어진 문자열이 주어졌을 때 치환할 수 있는 알파벳 문자열의 종류를 구하라. 06과 같이 묶는 건 안 된다. 


```
dp(i): s[:i+1] 까지의 substring에 대한 결과
dp(i) = dp(i-1) if s[i] is valid + dp(i-2) if s[i-1:i+1] is valid
지금 보는 포인트 기준으로, 현재 문자(s[i])가 유효하다면 dp[i-1] 을 만들 때 고려한 case들에서 그대로 연장할 수 있다. 
두 문자가 유효하다면(s[i-1:i+1]) dp[i-2] 에서 연장할 수 있다.
```


<details>

```python
def numDecodings(self, s: str) -> int:
    n = len(s)
    dp = [0] * (n+1)
    dp[-1] = 1  # 이 값을 1로 해줘야한다. dp[1] += dp[-1] if s[0:2] is valid 할 때 사용된다.

    if 1 <= int(s[0]) <= 9:
        dp[0] = 1

    for i in range(1, n):
        if 1 <= int(s[i]) <= 9:
            dp[i] += dp[i-1]
        if 10 <= int(''.join(s[i-1:i+1])) <= 26:
            dp[i] += dp[i-2]
    
    return dp[n-1]
```

근데 최근 두 개만 쓰니까 constant space로도 할 수 있겠다.

솔루션은 같은 논리인데 코드가 더 간단한다.

```python
def numDecodings(self, s: str) -> int:
    if s[0] == "0":
        return 0

    two_back = 1
    one_back = 1
    for i in range(1, len(s)):
        current = 0
        if s[i] != "0":
            current = one_back
        two_digit = int(s[i - 1: i + 1])
        if two_digit >= 10 and two_digit <= 26:
            current += two_back
        two_back = one_back
        one_back = current
    
    return one_back
```

</details>






### 198. House Robber

https://leetcode.com/problems/house-robber

문제: nums라는 리스트에 각 집에 저장되어 있는 돈의 양이 저장되어 있다. adjacent house를 털면 안 된다는 조건이 있을 때 최대로 털 수 있는 돈의 양을 구하라.

```
dp(i): nums[i] 까지 범위에서 최대한 얻을 수 있는 돈의 양
dp(i) = max(dp(i-1), dp(i-2) + nums[i])
```

<details>

```py
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * (n+1)
        dp[0] = nums[0]

        for i in range(1, n):
            dp[i] = max(dp[i-1], nums[i] + dp[i-2])

        return dp[n-1]
```

직전 값들만 저장함으로써 공간 최적화를 할 수 있다.

</details>








### 740. Delete and Earn

https://leetcode.com/problems/delete-and-earn

문제: 숫자 리스트가 있고 거기서 x를 뽑으면 그 값을 얻는다. 대신 그 x 하나를 지우고 모든 x+1와 모든 x-1를 지운다. 이 작업을 반복했을 때 얻을 수 있는 최대 sum은?
예시로, nums가 `[3,4,2]`면 답은 6, nums가 `[2,2,3,3,3,4]` 면 답은 9이다.

문제에 대한 이해도를 높여보자.    
어떤 값을 고르게 되면 그 양 옆은 아예 못 쓰게 된다. 그 말인 즉, 골랐던 값이 지워질 일은 없다는 뜻이기도 하다.    
그러면 문제를 house robber 로 재구성할 수 있다. x를 취하게 되면 모든 x를 취하게 되고, 모든 x-1와 모든 x+1를 버리게 되므로 x의 합을 index x의 값으로 둘 수 있다.    
그렇게 했을 때 그 리스트에서 연속으로 두 개를 고를 수 없을 때 최대의 sum을 구하는 문제가 된다.

<details>

```py
    def deleteAndEarn(self, nums: List[int]) -> int:
        targets = [0] * (max(nums) + 1)
        for num in nums:
            targets[num] += num
    
        # dp(i) = max(dp(i-1), dp(i-2) + targets[i])
        one_before, two_before = 0, 0
        for i in range(len(targets)):
            cur = max(one_before, two_before + targets[i])
            one_before, two_before = cur, one_before
        return max(one_before, two_before)

```

</details>







### 300. Longest Increasing Subsequence

https://leetcode.com/problems/longest-increasing-subsequence

문제: Given an integer array nums, return the length of the longest strictly increasing subsequence. 리스트 안에서 연속되지 않아도 된다. 오른쪽으로 가는 순서대로만 있으면 된다. 

TLE 각오하고 그냥 짰는데 beat 63% 나왔다. 근데 이게 dp solution이었다.   
memo라는 리스트를 만들어서 1로 초기화한다. memo[i]는 nums[:i+1]의 범위에서 nums[i]가 골라졌을 때의 longest increasing subsequence 길이다.   
그러면 왼쪽부터 차례대로 이동하면서 memo[i]를 `(0~i-1의 memo 값 중 최대) + 1`로 업데이트하면서 간다. 이 때, nums[j]가 nums[i]보다 작지 않으면 무시해야한다.   

O(N^2) / O(N)

<details>

```py
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        memo = [1] * n

        for i, num in enumerate(nums):
            tmp = -math.inf
            for j in range(i):
                if nums[j] >= num:
                    continue
                tmp = max(tmp, memo[j])
            memo[i] = max(tmp + 1, memo[i])
        
        return max(memo)
```

</details>


greedy with binary search    
이 방법은 볼 때마다 새롭다.   
왼쪽부터 차례대로 subsequence를 만든다. 계속 이어가다가 다음 숫자 x가 subsequence의 last element보다 작다면 x를 더 붙일 수 없다.   
그 상황에서 새로운 array를 만들어서 기존 subsequence에서 x보다 작은 부분을 넣고 그 다음에 x를 넣을 수 있다.   
이런 식으로 뒤에 못 붙이는 수가 나올 때마다 array를 새로 만들어가다가 다 끝나면 그 중 가장 긴 array 길이를 반환할 수 있다.

하지만 이 방법은 최적화가 가능하다. 최대한 길게 만들어야하고 길이만 중요하니까 하나의 array를 같이 쓸 수 있다.   
새로운 x가 나왔는데 array 뒤에 못 붙인다면 x를 array에서 맞는 자리로 넣어준다.   
array는 sorted 상태이기 때문에 binary search를 사용할 수 있다.   
이렇게 하나의 array를 업데이트한 뒤 마지막에는 그 array의 길이를 반환하면 된다.    


<details>

```py
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        res = [nums[0]]
        for i in range(1, n):
            num = nums[i]
            if num > res[-1]:
                res.append(num)
                continue
            target_idx = bisect_left(res, num)
            res[target_idx] = num
        return len(res)
            
```

O(N logN) / O(1)    
기존 list를 업데이트하면 O(1)도 가능하다.   

근데 bisect_right하면 왜 실패하는지 모르겠다.

</details>









### 322. Coin Change

https://leetcode.com/problems/coin-change/

문제: coins 라는 리스트에는 서로 다른 종류의 coin이 있다. amount라는 값이 주어졌을 때 coins의 coin으로 amount를 make up할 수 있는 최소의 coin 수를 구하라.

x라는 양은 x-coin에서 coin 하나를 더 쓰면 만들 수 있다. 이 관계를 이용하여 len(coins) 번 iterate한다.

```
dp[i]: Minimum number of coins to make up to i
dp[i] = min(dp[i-k] for k in coins)
```

Time Complexity: amount * len(coins)

<details>


```py
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [math.inf] * (amount+1)
        dp[0] = 0

        for i in range(amount+1):
            for coin in coins:
                if i-coin < 0:
                    continue  # coins를 맨 처음에 sort하고 여기서는 break 해버리면 조금 최적화가 된다.
                dp[i] = min(dp[i], dp[i-coin] + 1)
        
        if dp[amount] == math.inf:
            return -1
        return dp[amount]
```

</details>


















## Grid Problems





### 63. Unique Paths II

https://leetcode.com/problems/unique-paths-ii/

문제: robot이 m x n matrix의 제일 왼쪽 위에 놓여져있고 오른쪽이나 아래로만 움직일 수 있다. matrix[i][j]의 값이 1이라면 그 곳은 로봇이 움직일 수 없다. grid의 제일 오른쪽 아래에 갈 수 있는 경로의 수를 구하라.


```
dp(i, j): Number of the unique paths from (0, 0) to (i, j)
dp(i, j) is 
- 0 if (i, j) is out of the grid
- 0 if (i, j) is an obstacle
- 1 if (i, j) is the top-left corner
- dp(i-1, j) + dp(i, j-1) otherwise
```



<details>

top down

```python
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if  obstacleGrid[0][0] == 1:  # 이 부분을 빠뜨리지 말자. 답도 틀리고 수행 시간도 길어진다.
            return 0

        @lru_cache(maxsize=None)
        def getPathHelper(i, j):
            if i == 0 and j == 0:
                return 1
            if not (0 <= i < m and 0 <= j < n):
                return 0
            if obstacleGrid[i][j] == 1:
                return 0
            return getPathHelper(i-1, j) + getPathHelper(i, j-1)
        return getPathHelper(m-1, n-1)
```

bottom up

```python
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        if  obstacleGrid[0][0] == 1:  # 이 부분을 빠뜨리지 말자. 답도 틀리고 수행 시간도 길어진다.
            return 0

        prev_row = [0] * n
        prev_row[0] = 1

        for i in range(m):
            cur_row = [0] * n
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    cur_row[j] = 0
                    continue
                cur_row[j] = cur_row[j-1] + prev_row[j]
            prev_row = cur_row
        
        return prev_row[n-1]
```

</details>









### 64. Minimum Path Sum

https://leetcode.com/problems/minimum-path-sum/description/

문제: m x n grid에서 non-negative 숫자로 채워져있다. top left에서 right bottom으로 가야하는데 오른쪽 혹은 아래로만 움직일 수 있다. 가는 길에 있는 숫자의 합이 최소가 되도록 가라.




```
dp(i, j): The minimum sum of costs to reach (i, j) from the top left corner
dp(i, j) is 
 - grid[i][j] if i == 0 and j == 0
 - inf if (i, j) is out of the grid
 - min(dp(i-1, j), dp(i, j-1)) + grid[i][j] otherwise
```

- top down: recursion 방식. O(mn) / O(mn)    
- bottom up: iterative하게 하려면 하나의 row를 저장하면서 하는 방법이 있다. O(mn) / O(n)    
- optimization on space: bottom up을 하면서 original matrix에 업데이트하는 방법도 있다. O(mn) / O(1)

<details>

top down

```py
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        @lru_cache(maxsize=None)
        def getPathHelper(i, j):
            if i == 0 and j == 0:
                return grid[i][j]
            if not (0 <= i < m and 0 <= j < n):
                return math.inf
            return min(getPathHelper(i-1, j), getPathHelper(i, j-1)) + grid[i][j]
        
        return getPathHelper(m-1, n-1)
```

bottom up with updating the original grid

```py
    def minPathSum(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])

        for i in range(m):
            for j in range(n):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] = grid[i][j-1] + grid[i][j]
                elif j == 0:
                    grid[i][j] = grid[i-1][j] + grid[i][j]
                else:
                    grid[i][j] = min(grid[i-1][j], grid[i][j-1]) + grid[i][j]
        return grid[m-1][n-1]
```



</details>










### 221. Maximal Square

https://leetcode.com/problems/maximal-square/

문제: mxn binary matrix가 0 혹은 1로 채워져있다. 1로만 이루어진 가장 큰 정사각형의 넓이를 반환하라.


- 어떤 꼭지점 (i,j) 를 기준으로 왼쪽, 위, 왼쪽위 점들이 둘러싸는 점들이다.
- 왼쪽 점 (i, j-1), 위쪽 점 (i-1, j) 이 겹치는 부분은 현재 점을 기준으로도 연장될 수가 있다.
- 만약, 4, 4 라면 현재 점 기준으로 왼쪽 4개, 위쪽 4개를 더 포함할 수 있다는 건데, 제일 왼쪽 위 꼭지점은 아직 알 수 없다.
- (i-1, j-1) 도 만약 4라면 제일 왼쪽 위 꼭지점도 포함한다는 뜻이다. 왜나하면 바로 왼쪽 점인 (i, j-1) 과 동일하게 왼쪽으로 뻗어나가는데 한 칸 위까지 뻗어나가기 때문이다.

```
dp(i, j): matrix[i][j] 위치를 오른쪽 아래 꼭지점으로 두어서 왼쪽 위로 만들 수 있는 최대의 정사각형의 한 변 길이    
dp(i, j) = min(dp(i-1,j), dp(i,j-1), dp(i-1,j-1)) + 1 
```

이렇게 하면 dp로 풀이는 가능하고, 공간 최적화를 하려면 직전 row의 정보만 보관하면 된다.


<details>

```py
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        n_row = len(matrix)
        n_col = len(matrix[0])
        
        prev_row = [0] * n_col
        cur_row = [0] * n_col

        max_side = 0

        for i in range(n_row):
            for j in range(n_col):
                if matrix[i][j] != '1':
                    continue
                if j == 0:
                    cur_row[j] = 1  # 여기서 max_side 업데이트 하지 않고 continue로 넘어가버려서 틀렸었다. 
                else:
                    cur_row[j] = min(cur_row[j-1], prev_row[j], prev_row[j-1]) + 1
                max_side = max(max_side, cur_row[j])
            prev_row = cur_row  # 밑에서 cur_row가 바라보는 객체를 다시 만들어주니까 deepcopy 없이 그냥 prev_row가 바라보는 객체만 바꿔주면 된다.
            cur_row = [0] * n_col
        
        return max_side * max_side
```

</details>

prev_row, cur_row 두 개를 쓰는 게 아니라 prev_row, left_value 이렇게 두 개를 쓰려고 해봤다.    
그런데 row를 오른쪽으로 이동하면서 prev_row의 자기 위치를 업데이트해야하는데 그렇게 하면 (i-1, j-1) 위치를 구하기가 어렵다.     
왜냐하면 prev_row[j-1]은 left_value와 동일하기 때문이다.    
그냥 row 두 개를 쓰자.   








### 931. Minimum Falling Path Sum


https://leetcode.com/problems/minimum-falling-path-sum/description/

문제: n x n matrix가 있을 때 falling path 중 minimum sum을 구하라. falling path란 제일 윗 row에서 제일 밑 row 까지 내려오는데 내려올 때 바로 아래나 대각선 아래로만 내려오는 path를 의미한다.



```
dp(i, j): minimum falliing path sum to get to matrix[i][j]
dp(i, j) = matrix[i][j] + min(dp(i-1, j-1), dp(i-1, j), dp(i-1, j+1)))
```

grid 문제를 연속으로 푸니까 기본 문제는 똑같은 틀에서 벗어나질 않네.


이것도 마찬가지로 bottom up으로 할 수 있는데 O(N) space로 할 수 있다. 그냥 밑에 row부터 차례대로 올라오는 것이다. 결국 row 0 의 결괏값만 알면 되는 건데 이는 row 1의 결괏값만 필요하다.

<details>

```py
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        n = len(matrix)

        @lru_cache(maxsize=None)
        def helper(i, j):
            if not (0 <= i < n and 0 <= j < n):
                return math.inf
            if i == 0:
                return matrix[i][j]
            return matrix[i][j] + min(helper(i-1, j-1), min(helper(i-1, j), helper(i-1, j+1)))
        
        res = math.inf
        for j in range(n):
            res = min(res, helper(n-1, j))
        return res
```

</details>


---








