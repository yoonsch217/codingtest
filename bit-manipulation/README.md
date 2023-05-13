## 개념

### 배경 지식

base-X number 란 X만큼 증가했을 때 carry가 생기는 걸 말한다.   

decimal은 base-10인데 decimal value를 non-decimal base-X value로 변환할 때는 X로 계속 나누면 된다.   
몫이 0이 아닐 때까지 계속 X로 나눈다. 그러면 각각의 나누기에서 나온 나머지(remainder)들을 거꾸로 읽으면 된다.   
소수점 자리의 경우(fractional part)는 X로 곱하면서 나온 값에서 interger part를 순서대로 읽으면 된다.   

non decimal 인 두 base 의 수끼리 convert하는 경우는 decimal을 거친 뒤 가는 방법이 있다.   
그러나 base-2 와 base-8의 수 사이의 convert의 경우는 굳이 decimal로 거치지 않아도 된다.   
base-2의 숫자 세 개마다 base-8의 숫자로 바뀌게 된다.

8bit signed integer의 경우 가장 높은 자리는 부호를 나타낸다.   
-2의 경우는 machine number로 10000010 로 나타낼 수 있다.   
10000010(2) 가 원래 나타내는 값은 130이 되지만 컴퓨터에서는 -2로 처리가 된다. -2를 true value, truth value라고 한다.   

- original code: 숫자를 2진수로 나타낸 값이다.
- inverse code: non negative int의 경우는 original code와 동일하다. negative int의 경우는 origincal code에서 sign bit 빼고 다 flip한 값이다.
- complement code: non negative int의 경우는 origincal code와 동일하다. negative int의 경우는 inverse code에서 1을 더한 값이다. complement의 크기를 구하기 위해서는 전체를 다 flip하고 1을 더하면 된다.

original code의 경우 두 가지 문제점이 있다.   
0은 +0, -0 두 값이 동일하기 때문에 동일한 수를 다르게 표현할 수 있다.   
original code로 뺄셈을 하면 값이 틀리다.   
반면 complement code의 경우는 8-bit 기준으로 -128을 10000000 으로 표현할 수가 있다.   
10000000 라는 complement notation을 보면 우선 음수이고, magnitude를 보기 위해 전체 flip 후 1을 더하면 10000000 = 128이 된다.    
따라서 컴퓨터 연산에서는 complement code를 사용하게 된다.

- 64 라는 decimal value의 표현 => 01000000 / 01000000 / 01000000
- -64 => 11000000 / 10111111 / 11000000

### Bitwise operation

bit operation

- binary operation
  - AND, OR, XOR
- shift operation
  - left shift, right shift
- unary operation
  - negation

XOR(`^`)는 각 대응하는 bit가 같으면 0이고 다르면 1이다. 각각의 값은 상관 없이 두 값이 같은지 다른지만 본다. 이게 AND와의 차이이다.   

negation(`~`)은 각 bit를 flip 하는 연산이다. 참고로 complement notation 기준인 것 같다.    
5의 경우 00000101 인데 negate 하면 11111010 이 된다. 이 값은 complement notation이고 음수이므로 magnintude를 구하기 위해서는 전체 flip 후 1을 더해야한다. 따라서 -6이 된다.   
즉 negation은 부호를 바꾸고 magnitude에 1을 더한 값이 된다.   

shift에는 arithmetic shift와 logical shift가 있다.   
`<<` left shift opertion의 경우는 bit를 왼쪽으로 옮기면서 high bit는 버리고 low bit는 0으로 채운다. arithmetic shift나 logical shift나 동작이 동일하다.   
`>>` right shift operation의 경우는 반대 방향으로 shift하면서 low bit를 버린다. arithmetic shift의 경우는 highest bit로 high bit를 채우고 logical shift는 0으로 채운다. arithmetic은 부호를 유지하는 operation이다.   
c++에서 unsigned의 경우는 right shift를 하면 logical shift를 하게 되고 signed의 경우는 arithmetic shift를 하게 된다.   

shift operation을 사용하면 multiplication 과 division을 효율적으로 할 수 있다.   
left shift는 곱셈에 사용될 수 있다.   
어떤 값 a를 left shift n 번을 하면 `a * 2^n` 이 된다. shift 할 때마다 각 숫자에 2를 곱하는 것과 동일하기 때문이다.(결합법칙 생각)    
`a * 6`을 하고 싶다면 `a*4 + a*2` 이므로 두 번 shift 한 값과 한 번 shift 한 값을 더하면 된다.   
right shift의 경우는 동일하게 `a / 2^n` 의 값이 된다. rounded down 된 값을 얻게 된다.   


AND, OR, XOR, negation 연산에 대해서는 다음과 같은 properties가 있다.   
- Idempotent law (note that XOR does not satisfy the idempotent law)
- Commutative law
- Associativity
- Distributive Law
- De Morgan's Law
- -a = ~(a-1)

