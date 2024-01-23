### prime set

주어진 수 n에 대해서 소수의 약수를 구하는 함수이다.   
예를 들어 30이라는 수가 주어질 때, 처음에 2로 나눠지니까 2를 추가하고 이미 구한 divisor 2는 빼준 뒤 prime_set(15)를 구한다.   


```py
def primes_set(self,n):
    for i in range(2, int(math.sqrt(n))+1):
        if n % i == 0:
            return self.primes_set(n//i) | set([i])
    return set([n])
```    
