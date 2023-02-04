### 150. Evaluate Reverse Polish Notation

https://leetcode.com/problems/evaluate-reverse-polish-notation

문제: reverse polish notation으로 표현된 string에 대해서 그 결과를 반환하라. 

간단하다. 쭉 iterate하면서 연산자가 아니면 stack push하고 연산자면 최근 두 개 pop 한 뒤 계산하면 된다.

lambda를 쓰면 다르게 풀 수도 있다.   

<details>
  
```python
def evalRPN(self, tokens: List[str]) -> int:
        
    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "/": lambda a, b: int(a / b),
        "*": lambda a, b: a * b
    }
    
    stack = []
    for token in tokens:
        if token in operations:
            number_2 = stack.pop()
            number_1 = stack.pop()
            operation = operations[token]
            stack.append(operation(number_1, number_2))
        else:
            stack.append(int(token))
    return stack.pop()
```

</details>
