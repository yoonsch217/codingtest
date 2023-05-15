### 405. Convert a Number to Hexadecimal

https://leetcode.com/problems/convert-a-number-to-hexadecimal/

문제: int가 주어졌을 때 hexadecimal number 를 나타내는 string을 반환하라. 음수는 complement code 로 구현이 되어 있다. input이 26이면 output은 "1a"이다. leading zero는 없앤다. int는 4바이트이다. 

binary로 변환한 뒤 base 16으로 변환하는 개념이 있다.   
input 을 2비트로 간주하고 4자리씩 확인을 해보면 된다.   
그러면 2비트로 변환한 후 오른쪽부터 네 개씩 끊어서 base 16으로 변환하면 되는데 이걸 bit operation으로 하면 편하다.

```python
    def toHex(self, num: int) -> str:
        if num == 0:
            return '0'
        
        hexes = '0123456789abcdef'
        
        outputs = []
        for i in range(8):
            cur = num & 15
            outputs.append(hexes[cur])
            num = num >> 4
        outputs.reverse()
        return ''.join(outputs).lstrip('0')
```




### 190. Reverse Bits

https://leetcode.com/problems/reverse-bits/description/

문제: 32bit unsigned integer를 reverse한 값을 구하라.

나는 맨 뒷자리부터 보면서 맨 앞으로 보내고 싶었다. 그래서 2로 나눈 나머지에 차례대로 가장 큰 bit 위치(2^31) 부터 차례대로 곱하면서 더해줬다.

```python
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range(31, -1, -1):
            res += pow(2, i) * (n % 2)
            n = n >> 1
        return res
```

solution에 이런 것도 있는데 우선 넘어가자.


```py
    def reverseBits(self, n):
        n = (n >> 16) | (n << 16)
        n = ((n & 0xff00ff00) >> 8) | ((n & 0x00ff00ff) << 8)
        n = ((n & 0xf0f0f0f0) >> 4) | ((n & 0x0f0f0f0f) << 4)
        n = ((n & 0xcccccccc) >> 2) | ((n & 0x33333333) << 2)
        n = ((n & 0xaaaaaaaa) >> 1) | ((n & 0x55555555) << 1)
        return n
```




### 371. Sum of Two Integers

https://leetcode.com/problems/sum-of-two-integers/description/

문제: integer a, b 가 주어졌을 때 `+`, `-` 연산 없이 두 integer의 합을 구하라.

XOR 연산을 하면 carry 를 무시한 합을 구할 수가 있다. bit 상태에서 다르다는 건 0, 1 이니까 XOR의 결과인 1이 나오게 되고, XOR 결과가 0이 나오는 경우는 0, 0 이거나 1, 1이다.   
이제 carry를 구해야한다. 1, 1 일 때 carry가 생기는 것이기 때문에 `(x&y) << 1` 로 구할 수가 있다.   

뺄셈도 구할 수가 있다.   
XOR의 결과는 difference of two integers without taking borrow into account 이다.   
borrow는 `((~x) & y) << 1` 로 구할 수 있다.   

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


어렵고 신기하다..

32 bit로 정해져있기 때문에 O(1) / O(1) 이다.





### 136. Single Number

https://leetcode.com/problems/single-number

문제: int 리스트가 있는데 하나의 수 빼고는 다 두 개씩 존재한다. 하나만 존재하는 수를 구하라. constant space.

XOR 연산을 하면 동일한 숫자는 사라지게 된다.

```py
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res = res ^ num
        return res
```





### 1349. Maximum Students Taking Exam

https://leetcode.com/problems/maximum-students-taking-exam/



