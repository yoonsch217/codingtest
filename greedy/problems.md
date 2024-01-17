

### 881. Boats to Save People

https://leetcode.com/problems/boats-to-save-people/

문제: people 리스트가 주어지는데 각 값은 몸무게이다. 보트들에 사람을 최대 두 명 태울 수 있는데 `limit` 의 무게를 넘을 순 없다. 모든 사람을 태우기 위한 최소의 보트 수를 구하라.
limit 보다 무거운 사람은 없다.


먼저 정렬을 한다. 그러고 two pointers로 오른쪽부터 하나, 왼쪽부터 하나를 놓고 이동한다.   
오른쪽부터 무거운 사람을 태우면서 left pointer 몸무게가 더 태울 수 있는 무게면 태우고 안 되면 만다.    
right pointer 이동할 때마다 보트를 사용하는 거니까 결과값을 1 늘린다.

<details>

```py
    def numRescueBoats(self, people: List[int], limit: int) -> int:
        people.sort()
        l, r = 0, len(people) - 1
        cnt = 0
        while l < r:
            if people[r] + people[l] <= limit:
                l += 1
            r -= 1
            cnt += 1
        if l == r:
            cnt += 1
        
        return cnt
```

</details>

이 방법은 at most 두 명까지 태울 수 있어서 가능한 것 같다. 두 명 제한이 없고 weight sum limit만 있다면? 




### 279. Perfect Squares

https://leetcode.com/problems/perfect-squares/

문제: 어떤 int n이 주어졌을 때 perfect square로만 합쳐서 n을 만들도록 할 때의 perfect square number의 최소의 개수를 구하라. perfect square는 정수의 제곱이다.

dp에 정리해놨다.