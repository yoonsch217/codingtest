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

