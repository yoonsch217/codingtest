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

dp(i): 문자열s의 i ~ end까지 substring에 대한 결과(0 ~ i로 해도 된다.)    
dp(0)을 구하면 된다.   
dp(i)에 대해 j를 i+1부터 끝까지 이동시키면서 s[i:j]가 word_dict에 있는지 확인하고 있으면 dp(j)를 반환하면 된다.   
근데 substring 자르고 dp recursive하게 부를 때 index 설정하는 게 헷갈린다.   
```
dp(i): [0~i]까지의 substring에 대한 결과
dp(i) = dp(i-j) and exists(s[i-j+1 : i+1]) for j in range(1, i+2), dp(음수) = True
j range가 헷갈렸는데 dp(i-1) and exists(s[i:i+1]) 부터 dp(-1) and exists(s[0:i+1]) 가 나오도록 설정해야한다.
```

O(N^2) /  O(N)



### Maximal Square

https://leetcode.com/problems/maximal-square/

문제: mxn binary matrix가 0 혹은 1로 채워져있다. 1로만 이루어진 가장 큰 정사각형을 찾아라.

```
dp(i, j): matrix[i][j] 위치에서 왼쪽 위로 만들 수 있는 최대의 정사각형 크기라 정의한다.    
dp(i, j) = min(dp(i-1,j), dp(i,j-1), dp(i-1,j-1)) + 1 
```
먼저 dp(i-1, j), dp(i, j-1) 을 생각해서 겹치는 영역이 최대 정사각형의 후보이다. 두 개가 같은 경우는 모서리가 1인지 고려해야하는데 그 상황을 dp(i-1, j-1) 로 채운다고 접근하면 될 것 같다.   
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


### 518. Coin Change II

https://leetcode.com/problems/coin-change-ii

문제: coins 라는 리스트는 사용할 수 있는 coin 종류가 있고 amount라는 int가 있다. coins에 있는 coin으로 amount를 만드는 조합의 수를 구하라.

내 풀이   
```
dp(i, upper): Number of combinations to make up i with using coins not bigger than upper
dp(i, upper) = dp(i-coin, coin) for k in coins not greater than upper
dp(0) = 1
dp(i) = 0 if i < 0
```

number of coins = k 라고 할 때, O(k * (amount / min(coins))) 가 시간 복잡도 아닐까.    
amount / min(coins) 만큼 recursion 함수가 호출되고 각 recursion마다 len(coins) 만큼 iterate하니까?   


solution은 훨씬 간단하다. solution도 복잡도는 O(len(coins) * amount) / O(amount) 인데 실제 수행 시간은 훨씬 빠르다.

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
