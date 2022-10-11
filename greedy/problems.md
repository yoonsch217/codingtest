### 279. Perfect Squares

https://leetcode.com/problems/perfect-squares/

문제: 어떤 int n이 주어졌을 때 perfect square로만 합쳐서 n을 만들도록 할 때의 perfect square number의 최소의 개수를 구하라. perfect square는 정수의 제곱이다.

dp에 동일한 문제에 대한 풀이가 있다. dp보다 더 효율적인 방법이 greedy 방식이다.

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

