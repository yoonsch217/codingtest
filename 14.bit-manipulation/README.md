## 개념

### Base-X Number

base-X number 란 X만큼 증가했을 때 carry가 생기는 걸 말한다.   

decimal은 base-10인데 decimal value를 non-decimal base-X value로 변환할 때는 X로 계속 나누면 된다.   
몫이 0이 아닐 때까지 계속 X로 나눈다. 그러면 각각의 나누기에서 나온 나머지(remainder)들을 거꾸로 읽으면 된다.   
소수점 자리의 경우(fractional part)는 X로 곱하면서 나온 값에서 interger part를 순서대로 읽으면 된다.    

ex) 11을 2진법으로    
divisor 2로 계속 나누면, quotient 5 remainder 1, quotient 2 remainder 1, quotient 1 remainder 0, quotient 0 remainder 1   
=>  remainder를 거꾸로 나열하면 1011   

ex) 3/8을 2진법으로    
divisor 2로 계속 곱하면, whole number part 0 fractional part 3/4, whole number 1 fractional 1/2, whole number 1 fractional 0   
=> whole number part를 순서대로 나열하면 0.011

non decimal 인 두 base 의 수끼리 convert하는 경우는 decimal을 거친 뒤 가는 방법이 있다.   
그러나 base-2 와 base-8의 수 사이의 convert의 경우는 굳이 decimal로 거치지 않아도 된다.   
base-2의 숫자 세 개마다 base-8의 숫자로 바뀌게 된다.


### 8-bit Signed Integer

8bit signed integer의 경우 가장 높은 자리는 부호를 나타낸다.   
-2의 경우는 machine number로 10000010 로 나타낼 수 있다.   
10000010(2) 가 원래 나타내는 값은 130이 되지만 컴퓨터에서는 -2로 처리가 된다. -2를 true value 혹은 truth value라고 한다.   

- original code: 그냥 나타낸 코드이다. sign bit 포함이다.
- inverse code
   - non negative int: original code와 동일하다.
   - negative int: origincal code에서 sign bit 빼고 다 flip한 값이다.
- complement code
   - non negative int: origincal code와 동일하다. 
   - negative int: inverse code에서 1을 더한 값이다. 
   - two's complement: 어떤 positive integer에 대해서 음수 값을 구하려면, non-sign bits끼리 양수값 + 음수값이 10000000이 되어야 한다.   
   ex) +18 = 00010010, -18 = 11101110
   positive integer에서 flip을 한 값을 생각해보자. 그 둘을 합치면 항상 1111111이 된다. 우리는 10000000을 만들어야하므로 거기에 1만 더 하면 된다.    
   즉, positive integer의 flip한 값에서 1을 더한 게 complement code에서의 negative int의 크기이고 거기에 
   sign bit만 더해주면 된다.   
   음수값의 magnitude를 보려면 non-sign bits들을 전체 flip하고 1을 더해주면 된다.
   - 나만의 정리(or proof)
      - 음수-1 의 flip이 양수, 즉 magnitude이다. 그리고 flip끼리의 합은 항상 일정
      - 어떤 값 a에 대해서 (a-1).flip vs a.flip + 1 이 같다는 걸 증명해야한다.
      - (a-1).flip + a - 1 = A    
      (a-1).flip = A - a + 1    
      a.flip + a = A    
      a.flip + 1 = A - a    
      Therefore, a.flip + 1 = (a-1).flip - 1

예시

- 64 original code, inverse code, complement code => 01000000 / 01000000 / 01000000
- -64 original code, inverse code, complement code => 11000000 / 10111111 / 11000000
- -1 complement code: 1의 inverse는 11111110, 1 더하면 11111111

original code의 경우 두 가지 문제점이 있다.   
0은 +0, -0 두 값이 동일한데 동일한 수가 다르게 표현될 수 있어서 비효율적이다.(10000000, 00000000)    
original code로 뺄셈을 하면 값이 틀리다.   

반면 complement code의 경우는 8-bit 기준으로 -128을 10000000 으로 표현할 수가 있다.   
10000000 라는 complement notation을 보면 우선 음수이고, magnitude를 보기 위해 전체 flip 후 1을 더하면 10000000 = 128이 된다.    
따라서 컴퓨터 연산에서는 complement code를 사용하게 된다.









### Bitwise operation

binary operation
- AND(`&`)
- OR(`|`)
- XOR(`^`)
   - XOR는 각 대응하는 bit가 같으면 0이고 다르면 1이다. 각각의 값은 상관 없이 두 값이 같은지 다른지만 본다. 이게 AND와의 차이이다.   

unary operation
- negation(`~`)
   - negation은 각 bit를 flip 하는 연산이다.    
   - 5의 경우 00000101 인데 negate 하면 11111010 이 된다. 
   complement notation에서 이 값은 음수이므로 magnintude를 구하기 위해서는 전체 flip 후 1을 더해야한다. 따라서 negated five는 -6이 된다.  
   - 즉 양수에 대한 negation은 부호를 바꾸고 magnitude에 1을 더한 값이 된다.   

shift operation

- left shift (`<<`)
   - left shift opertion의 경우는 bit를 왼쪽으로 옮기면서 high bit는 버리고 low bit는 0으로 채운다. arithmetic shift나 logical shift나 동작이 동일하다.   
   - 숫자 a를 n번 shift하면 `a * 2^n`가 된다.
   - 곱셈에 사용될 수 있다. `a * 6`을 하고 싶다면 `a*4 + a*2` 이므로 두 번 shift 한 값과 한 번 shift 한 값을 더하면 된다.
- right shift (`>>`)
   - right shift operation의 경우는 반대 방향으로 shift하면서 low bit를 버린다. arithmetic shift의 경우는 highest bit로 high bit를 채우고 logical shift는 0으로 채운다. 
   - 숫자 a에 대해 n번 shift하면 `a // 2^n`가 된다.
- arithmetic shift는 부호를 유지하는 operation이다.   
   - c++에서 unsigned의 경우는 right shift를 하면 logical shift를 하게 되고 signed의 경우는 arithmetic shift를 하게 된다.   
- shift operation을 사용하면 multiplication 과 division을 효율적으로 할 수 있다.   



AND, OR, XOR, negation 연산에 대해서는 다음과 같은 properties가 있다.   
- Idempotent law (note that XOR does not satisfy the idempotent law)
- Commutative law
- Associativity
- Distributive Law
- De Morgan's Law

유용한 연산들
- Get i-th bit of the given number.: `num & (1<<i)`
- Set i-th bit to 1: `num | (1<<i)`  i-th만 1인 숫자와 or 연산을 하면 된다.
- Clear i-th bit: `num & (~(1<<i))` i-th 빼고 다 1인 숫자와 and 연산을 하면 된다.
- -a = ~(a-1)
- n & (n-1) 을 하게 되면 n에 존재하는 1 중에 가장 낮은 자리에 있는 1만 0으로 바꿔준다. 이걸 사용하면 어떤 binary에 있는 1의 수를 빠르게 셀 수 있다. 전체 리스트를 iterate할 필요 없이 1의 개수만큼만 iterate하면 된다. 
operation은 O(1)의 시간에 되나?
- `0 ^ x = x`, `x ^ x = 0` 
- https://leetcode.com/problems/sum-of-two-integers/solutions/84278/a-summary-how-to-use-bit-manipulation-to-solve-problems-easily-and-efficiently/


bit manipulation 문제에서 어떻게 시작할지 모르겠다면 input data에 대해 XOR 연산을 우선 해보면 힌트를 얻을 수도 있다.











# 응용

### Bit Mask

정수 하나만 이용해서 리스트를 사용한 것과 같은 효과를 낼 수 있다.     
예를 들어 다섯 개의 아이템이 있고 각각의 taken or not 에 대한 상태를 저장한다고 할 때, 10110(2) = 22 라는 숫자로 1, 3, 4 item은 taken, 2, 5 는 not taken 이라는 상태를 저장할 수 있다.    

- 모든 bit를 0으로 초기화: 0
- 모든 bit를 1로 초기화: -1
- i번째 원소 삭제: `num &= ~(1<<i)`
- i번째 원소 추가: `num |= (1<<i)`
- i번째 원소 확인: `num & (1<<i) >= 1`
- i번째 원소 toggle(flip): `num ^= (1<<i)`
- 가장 끝에 있는 1 구하기: `num & -num`, -num은 num을 flip한 거에 1을 더한 것이다. 
num의 마지막 0들을 1로 바뀌고, 거기에 1을 더하면 결국 num의 가장 낮은 자리의 1만 남게 된다. 
- 가장 끝에 있는 1을 0으로 바꾸기: `num &= (num-1)`






### State Compression via Bit Manipulation


두 가지의 상태가 있고 n 개의 아이템이 있을 때, 총 2^n 개의 상태가 필요하다. 보통 n이 20 이하일 때 state compression을 사용한다.   

DP에서 반복 연산을 피하기 위해 state compression이 사용되기도 한다. State Compression Dynamic Programming 이라고 하고 dp 문제 중에서 가장 어렵다.   
state definition 을 위해 state compression을 해야한다.   
어떻게 state를 표현할지와 state 들 사이의 관계를 알아내는 것이 중요하다.   

예시) ### 1125. Smallest Sufficient Team


