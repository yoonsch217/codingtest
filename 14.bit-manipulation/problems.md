### 405. Convert a Number to Hexadecimal

https://leetcode.com/problems/convert-a-number-to-hexadecimal/

문제: int가 주어졌을 때 hexadecimal number 를 나타내는 string을 반환하라. 음수는 two's complement method 로 구현이 되어 있다. 
input이 26이면 output은 "1a"이고 -1이면 ffffffff이다. 
0일 때를 제외하고는 leading zero는 없어야한다. int는 4바이트이다. 

binary로 변환한 뒤 base 16으로 변환하는 개념이 있다.   
input 을 2비트로 간주하고 4자리씩 확인을 해보면 된다.   
그러면 2비트로 변환한 후 오른쪽부터 네 개씩 끊어서 base 16으로 변환하면 되는데 이걸 bit operation으로 하면 편하다.


<details>

```py
    def toHex(self, num: int) -> str:
        if num == 0:
            return '0'
        hexnum = '0123456789abcdef'

        # 2^32개의 수가 있으니까 4byte = 32bit이다. 
        # 2^32 = 16^8 니까 hexadecimal로는 8자리가 된다.
        res = []
        for i in range(8):
            # 아래 4bit만 뽑으려면 1111과의 & 연산을 해야한다.
            cur = num & 15
            res.append(hexnum[cur])  # 가장 낮은 숫자가 리스트의 앞에 오게 된다.
            num = num >> 4  # 낮은 자리의 4 bit shift
        res.reverse()
        return ''.join(res).lstrip('0')
```

</details>










### 190. Reverse Bits

https://leetcode.com/problems/reverse-bits/description/

문제: 32bit unsigned integer를 reverse한 값을 구하라.

처음에는 리스트를 만들어서 reversed bits를 저장하고 다시 그 list를 iterate하면서 결괏값을 계산했다.   
근데 한 번의 iteration에서 바로 결과를 더해주는 게 좋아보인다.

<details>

```python
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range(32):
            cur_bit = n & 1
            if cur_bit != 0:
                res += cur_bit * pow(2, 31-i)
            n = n >> 1
        return res
```

</details>

solution에 이런 것도 있는데 우선 넘어가자.

<details>

```py
    def reverseBits(self, n):
        n = (n >> 16) | (n << 16)
        n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8)
        n = ((n & 0xf0f0f0f0) >> 4) | ((n & 0x0f0f0f0f) << 4)
        n = ((n & 0xcccccccc) >> 2) | ((n & 0x33333333) << 2)
        n = ((n & 0xaaaaaaaa) >> 1) | ((n & 0x55555555) << 1)
        return n
```

</details>








### 136. Single Number

https://leetcode.com/problems/single-number

문제: int 리스트가 있는데 하나의 element만 한 개 존재하고 나머지는 다 두 개씩 존재한다. 하나만 존재하는 수를 구하라. constant space. 
`[1, 1, 2, 2, 3]` => 3

XOR 연산을 하면 동일한 숫자는 사라지게 된다.

<details>

```py
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res = res ^ num
        return res
```

</details>








### 371. Sum of Two Integers

https://leetcode.com/problems/sum-of-two-integers/description/

문제: integer a, b 가 주어졌을 때 `+`, `-` 연산 없이 두 integer의 합을 구하라.


덧셈   
XOR 연산을 하면 carry 를 무시한 합을 구할 수가 있다. 두 bit가 같으면 그 자릿수는 0이고 다르면 1이다.     
이제 carry를 구해야한다. 1, 1 일 때 carry가 생기는 것이기 때문에 `(x&y) << 1` 로 구할 수가 있다.   
carry를 더하면서 또 carry가 생길 수 있으니까 carry가 안 생길 때까지 반복해야한다.

뺄셈   
마찬가지로 XOR의 결과가 difference of two integers without taking borrow into account 이다.   
두 bit가 다르다면 1이 나오고 같으면 0이 된다.   
borrow는 첫 번째 bit가 0이고 두 번째 bit가 1일 때만 생긴다. 따라서 `((~x) & y) << 1` 로 구할 수 있다.   
borrow는 윗 자리에서 받아와야하기 때문에 이것도 left shift를 한다.   

그런데 input이 양수라는 보장이 없기 때문에 좀 더 복잡해진다. 뺄셈 연산도 만든다.   

sign이 다를 때 덧셈 뺄셈은 그냥 나머지로 하고 마지막에 부호만 붙이면 되나?
여러 케이스들을 손으로 써보면서 감을 잡아야할 것 같다. 
음수에 대해서 -1, -2, -3 이게 어떻게 증가하고 음수끼리의 연산은 어떻게 되는지.

<details>

```py
    def getSum(self, a: int, b: int) -> int:
        x, y = abs(a), abs(b)
        # ensure that abs(a) >= abs(b)
        if x < y:
            return self.getSum(b, a)
        
        # abs(a) >= abs(b) --> sign of a determines the sign of the answer
        sign = 1 if a > 0 else -1
        
        if a * b >= 0:
            # sum of two positive integers x + y. even when those two are both negative, we can just add them and put the sign at the end.
            while y:  # at first, y is abs(b). after than, y is the carry that is computed for an iteration.
                answer = x ^ y
                carry = (x & y) << 1
                x, y = answer, carry
        else:
            # difference of two integers x - y
            # where x > y
            while y:
                answer = x ^ y
                borrow = ((~x) & y) << 1
                x, y = answer, borrow
        
        return x * sign
```

</details>


어렵고 신기하다..

32 bit로 정해져있기 때문에 O(1) / O(1) 이다.









### 1125. Smallest Sufficient Team

https://leetcode.com/problems/smallest-sufficient-team/description/

DP with state compression

```
dp[skills] = skills를 만족하기 위한 최소 인원의 team
작업을 진행하다가 dp[s]에 대해 더 작은 인원의 team이 발견된다면 바꿔치기 한다.
이 부분이 좀 헷갈렸다. 더 작은 인원이 보이면 이전 팀 정보를 버리고 바꿔치기 해도 되나.
바꿔쳤는데 나중에 이전 값이 필요할 때가 있을까? 없다. key가 skill 목록이다보니까 어찌됐든 그 skill 목록을 만들기만 하면 된다.
사람 목록은 더 이상 중요하지 않아진다.
```


<details>

```py
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        skill_to_idx = {}
        for i, s in enumerate(req_skills):
            skill_to_idx[s] = i
        
        skills_to_team = {}  # key: bit masked skill list, value: smallest team(list of people)
        """
        0에 대해 초기화를 하지 않고 `if cur_skills not in {}: 추가`로 했는데 그러면 corner case에 걸린다. 
        0을 만드는 최소 team은 [] 여야하는데 cur_skills가 0인 사람이 있다면 최소 team이 한 명이 된다.
        그렇게 되면 모든 dict를 iterate할 때 처음 보는 cur_skills에 대해 그 team을 기준으로 expand하게 된다.
        """
        skills_to_team[0] = []  

        for p_idx, p_skills in enumerate(people):
            cur_skills = 0
            # construct bit masked skill for this person
            for skill in p_skills:
                if skill not in skill_to_idx:
                    continue
                skill_bit = (1 << skill_to_idx[skill])
                cur_skills |= skill_bit
            
            if cur_skills not in skills_to_team:  # skills_to_team[0] = [] 넣음으로써 필요없어졌다.
                skills_to_team[cur_skills] = [p_idx]
            
            for prev_skills, prev_team in dict(skills_to_team).items():  # dict 원본을 쓰면 size가 변해서 에러나니까 복사해서 iterate한다.
                updated_skills = prev_skills | cur_skills  # 이전과 지금을 합친 state
                if updated_skills == prev_skills:  # 바뀐 게 없다면 무시한다. 지금 p를 안 넣는 게 이득
                    continue
                if updated_skills not in skills_to_team:
                    skills_to_team[updated_skills] = prev_team + [p_idx]
                if len(skills_to_team[updated_skills]) > len(prev_team) + 1:
                    skills_to_team[updated_skills] = prev_team + [p_idx]
        
        return skills_to_team[(1 << len(req_skills)) - 1]  # 전체 skill에 대한 값을 반환한다.
```

- Time O(people * 2^skill)
- Space O(2^skill)

</details>








### 1349. Maximum Students Taking Exam

https://leetcode.com/problems/maximum-students-taking-exam/


https://leetcode.com/problems/find-the-shortest-superstring/description/
https://leetcode.com/problems/parallel-courses-ii/description/

