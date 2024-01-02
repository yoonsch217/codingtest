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



### Maximal Square

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



BFS 로 풀면 TLE가 안 난다.

- n 이하의 제곱수들을 구해서 저장한다.
- BFS의 한 level을 리스트 혹은 set으로 정의를 한다.
- 처음에는 root부터 시작이니까 {n} 가 초깃값이다.
- level의 각 원소마다 돌면서 제곱수라면 그때의 level을 반환한다.
- 제곱수가 아니라면 다음 level 리스트에 현재 보는 값에서 제곱수를 뺀 값을 넣어준다.
- 그 level의 작업이 끝나면 다음 level에 대해 작업해준다.


<details>

```py
    def numSquares(self, n: int) -> int:
        if n < 2:
            return n
        lst = []
        i = 1
        while i * i <= n:
            lst.append( i * i )
            i += 1
        cnt = 0
        toCheck = {n}
        while toCheck:
            cnt += 1
            temp = set()
            for x in toCheck:
                for y in lst:
                    if x == y:
                        return cnt
                    if x < y:  # 처음에는 x 값이 lst의 최솟값인 1보다 클 것이다. 그러다가 x가 y보다 작아진다면 이후 y보다도 다 작을 것이므로 더 볼 필요가 없다.
                        break
                    temp.add(x-y)
            toCheck = temp

        return cnt
```

</details>


greedy 방식을 사용할 수도 있다. `result = dp(n, k) for n in [i, .. n]` 이고 dp(n, k)는 k 개의 perfect square로 n을 만들 수 있으면 true를 반환하고 그게 그때의 최적의 답이다.   
`dp(n, k) = dp(n-squarenum, k-1) + 1`      
이걸 증명하는 건 contradiction을 이용할 수 있다. dp(n, i)가 있고 그 뒤에 dp(n, j)가 나왔고 dp(n, j)가 더 작은 수라고 하자. dp(n, j)의 답은 j인데 이는 i보다 작아야한다. 그런데 먼저 수행된 i가 더 작아야하므로 모순이다.   
Time Complexity: O(n^(h/2)) where h is the maximal number of recursion that could happen   
n-ary tree로 생각할 수 있다. 어떤 parent node의 숫자를 기준으로, 그 숫자보다 작은 square number를 뺀 node들을 child node로 갖는다.   

greedy 방식을 n-ary tree로 생각할 때, 각 레벨을 BFS로 탐색하는 것으로 볼 수도 있다.    
레벨이 곧 사용된 perfect square 숫자의 개수이기 때문이다.   







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

문제: 숫자 1부터 26은 각각 A부터 Z까지 매핑될 수 있다. 숫자로 이루어진 문자열이 주어졌을 때 치환할 수 있는 알파벳 문자열의 종류를 구하라. 06으로 묶는 건 불가능하다. 

언제나 dp(i)에 대한 일반식을 구하는 게 먼저이다.   
dp(i)는, 
- tmp = 0
- if s[i] is valid, tmp += dp(i-1)
- if s[i-1:i+1] is valid, tmp += dp(i-2)
- dp(k) = 1, where k < 0
- if tmp == 0, break and return 0

어떤 문자열에서 하나가 추가됐을 때, 유효하다면 그 결과는 바뀌지 않고 이전 결과가 그대로 된다. 
왜냐하면 추가됐을 때 valid한 조건이 그대로 유지되는 거지 뭔가가 경우의 수가 증가한 게 아니기 때문이다.   
이런 개념에서, dp(i)는 뒤에서부터 하나씩 짤라가며 s[i-k : i+1] 가 valid하면 dp(i-k-1)을 추가해준다.   
그런데 뒤에서부터 자를 때 세 개 이상 자르면 valid할 수가 없다. 최대 두자릿수이기 때문이다.   
그리고 0이 나오면 바로 break하도록 했는데 0이 나온 뒤에는 뭘 붙여도 valid할 수가 없기 때문이다.   

조금 설명하면서도 이상하긴한데.. 논리는 맞았다.


<details>

```python
def numDecodings(self, s: str) -> int:
    n = len(s)
    memo = [0] * (n+2)
    memo[-1] = 1
    memo[-2] = 1

    def is_valid(target: str) -> bool:
        if len(target) not in [1, 2]:
            return False
        if target[0] == '0':
            return False
        target_int = int(target)
        if target_int > 26:
            return False
        return True

    for i in range(n):
        tmp = 0
        if is_valid(s[i]):
            tmp += memo[i-1]
        if i > 0 and is_valid(''.join(s[i-1:i+1])):
            tmp += memo[i-2]
        if tmp == 0:
            break
        memo[i] = tmp
        print(tmp)
    
    return memo[n-1]
```

근데 constant space로도 할 수 있겠다. 최근 것만 쓰니까.

솔루션은 같은 논리인데 코드가 더 간단한다.

```python
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





### 918. Maximum Sum Circular Subarray


https://leetcode.com/problems/maximum-sum-circular-subarray

문제: circular integer array nums가 있다. 동일한 위치의 element가 두 번 이상 나오지 않도록 했을 때의 maximum possible sum of a non-empty subarray를 구하라. 

Kadane's Algorithm

두 가지로 나눌 수가 있다.
- array 안에 포함되는 subarray
- for i < j, index i에서부터 끝까지의 subarray 와 처음부터 j까지의 subarray를 연결한 array

첫 번째는 Kadane's algorithm으로 구할 수 있다.

두 번째는 또 여러 step으로 이루어진다. subarray를 이루는 두 개의 subarray가 하나는 맨 왼쪽이 index 0이고 하나는 맨 오른쪽이 index n-1임을 이용한다.
- rightMax(i)를 시작이 i 이후이고 끝이 n-1인 subarray들 중에 가장 sum이 큰 값으로 정의한다.
- ~leftMax(i)를 시작이 0이고 끝이 i 이전인 subarray들 중에 가장 sum이 큰 값으로 정의한다.~
- ~그러면 두 번째 케이스에 대해서는, i를 1부터 n-1까지 이동하면서 leftMax(i-1) + rightMax(i) 들 중 가장 큰 값이 된다.~
- leftMax를 할 필요가 없이 그냥 prefixSum으로 하는 게 더 간단하다. 
- i를 0부터 n-2까지 증가하면서 prefixSum을 업데이트한다. sepcialSum(i) = max(specialSum(i-1), prefixSum(i) + rightMax(i+1))

이렇게 첫 번째 case와 두 번째 case 의 결과 중 큰 값이 정답이다.   
O(N) time, O(N) space

<details>

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

이게 내 solution인데 앞에서부터 iterate하는 건 Kadane's algorithm이랑 circular 작업에서랑 둘 다 존재하고 있다.
이 두 작업을 같은 iteration에서 하는 게 더 효과적이다.



</details>

O(N) time, O(1) space를 사용하는 풀이법도 있다.

이 문제를 다시 생각해보면 하나의 array에서 max sum subarray를 찾는 건데 그게 한번에 이어져있든, 끝과 처음을 통해 이어져있든 상관 없는 것이다.   
한번에 이어진 경우는 여전히 기본 Kadane's algorithm으로 구할 수 있다.   
끝과 처음을 통해 이어진 경우를 다른 방식으로 풀 수 있는데, 전체 array에서 가운데 subarray를 뺀 것이라고 생각하면 된다.   
그러면 전체 array sum에서 minimum sum subarray를 구해서 그 부분을 빼면 된다.

- Kadane's algorithm을 통해 normal answer 구하기
- 전체를 더한 arraySum 구하기
- Kadane's algorithm을 변형하여 minimum sum subarray를 구하기
- arraySum - minSum 이 linked answer가 된다. 
- 그런데 만약 minSum이 arraySum과 같다면? 전체 array가 min sum subarray라면 그걸 뺀 subarray는 invalid하다. 그런 경우는 normal answer를 반환하면 된다.
- 근데 arraySum == minSum 이라면 arraySum - minSum은 0이 될 테지. 
arraySum이 양수라면, array에 최소 하나의 양수가 있다는 거다. 그런데 minSum이 동일하려면 그 양수 하나밖에 없어야한다. 음수가 있다면 그게 min sum이 될 테고, 다른 양수가 있다면 덜 합치는 게 minSum이 돼야한다. 양수가 하나만 있다면 그건 normal answer에서 다뤄지니까 linked answer는 무시할 수 있다. 
arraySum이 음수라면, minSum이 동일하려면 그 array에는 양수가 없어야한다. 왜냐하면 양수가 있다면 minSum은 그 양수를 포함하지 않을테고 그러면 arraySum != minSum이 되기 때문이다. 다 음수라면 그 값은 normal answer보다 클 수가 없다. normal answer은 그 중에서 가장 작은 값 하나만 골랐을 것이기 때문이다. normal answer이 음수가 되고 linked answer이 0이 되기 때문에 단순히 max(normal answer, linked answer) 하면 안 된다. 따라서 이 case에 대한 예외 처리를 해줘야한다.

<details>

```python
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



### 62. Unique Paths

https://leetcode.com/problems/unique-paths/description/

문제: robot이 m x n grid의 제일 왼쪽 위에 놓여져있고 오른쪽이나 아래로만 움직일 수 있다. grid의 제일 오른쪽 아래에 갈 수 있는 경로의 수를 구하라.


내 solution: top down

- starts from (0, 0), ends at (m-1, n-1)
- dp(i, j): Number of possible ways to reach the end when the robot is at (i, j) position.
- dp(i, j) = 0 if (i, j) is out of the grid, 1 if (i, j) is the target, dp(i+1, j) + dp(i, j+1) otherwise.

<details>

```python
        @lru_cache(maxsize=None)
        def getUniquePaths(i, j):
            if i == m-1 and j == n-1:
                return 1
            if not 0 <= i < m or not 0 <= j < n:
                return 0
            return getUniquePaths(i+1, j) + getUniquePaths(i, j+1)
        
        return getUniquePaths(0, 0)
```

</details>


내 solution: bottom up

- The answer for (i, j) position is the sum of the answer of (i+1, j) and the answer of (i, j+1)
- The answer for (m-1, n-1) position is 1.
- Starting from (m-1, n-1) position, moving left and when it reaches the leftend, go to the upper rightend and moving left again, it gets the current answer from the previous answer.


<details>

```python
prev_row = [0] * n
prev_row[-1] = 1
cur = 0
for i in range(m-1, -1, -1):
    for j in range(n-1, -1, -1):
        if j == n-1:
            cur = prev_row[j]
        else:
            cur += prev_row[j]
        prev_row[j] = cur

return cur
```

</details>

https://leetcode.com/problems/unique-paths-ii/description/ 이 문제도 있는데 obstacle만 추가된 문제이다. out of grid 조건에 is_obstacle 조건만 추가하면 된다.





### 64. Minimum Path Sum

https://leetcode.com/problems/minimum-path-sum/description/

문제: m x n grid에서 non-negative 숫자로 채워져있다. top left에서 right bottom으로 가야하는데 오른쪽 혹은 아래로만 움직일 수 있다. 가는 길에 있는 숫자의 합이 최소가 되도록 가라.


top down: recursion 방식. O(mn) / O(mn)

```
dp(i, j): i-th row, j-th col 에서 right bottom으로 가는 최단 cost
dp(i, j) = grid[i, j] + min(dp(i, j+1), dp(i+1, j))
dp(i, j) = inf if (i, j) is out of range, grid[i, j] if i == len(grid) - 1 and j == len(grid[0]) - 1
```

iterative하게 하려면 하나의 row를 저장하면서 하는 방법이 있다. O(mn) / O(n)

<details>

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n_row = len(grid)
        n_col = len(grid[0])

        prev_row = [math.inf] * n_col

        for i in range(n_row - 1, -1, -1):
            for j in range(n_col - 1, -1, -1):
                if j == n_col - 1:
                    if i == n_row - 1:
                        prev_row[j] = grid[i][j]
                    else:
                        prev_row[j] = grid[i][j] + prev_row[j]
                    continue
                prev_row[j] = grid[i][j] + min(prev_row[j], prev_row[j+1])
        
        return prev_row[0]
```

</details>

혹은 original matrix에 업데이트하는 방법도 있다. 이렇게 하면 O(1)인 것 같다.






### 931. Minimum Falling Path Sum


https://leetcode.com/problems/minimum-falling-path-sum/description/

문제: n x n matrix가 있을 때 falling path 중 minimum sum을 구하라. falling path란 제일 윗 row에서 제일 밑 row 까지 내려오는데 내려올 때 바로 아래나 대각선 아래로만 내려오는 path를 의미한다.


recursion 

```
dp(i, j): (i, j) 부터 시작해서 바닥까지 가는 minimum sum
dp(i, j) = matrix[i][j] + min(dp(i+1, j-1), dp(i+1, j), dp(i+1, j+1))
dp(i, j) = inf if (i, j) out of range, matrix[i][j] if i == n_row - 1
return max(dp(0, j))
```

참고로 out of range를 먼저 처리해줘야한다.

이것도 마찬가지로 bottom up으로 할 수 있는데 O(N) space로 할 수 있다. 그냥 밑에 row부터 차례대로 올라오는 것이다. 결국 row 0 의 결괏값만 알면 되는 건데 이는 row 1의 결괏값만 필요하다.


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
