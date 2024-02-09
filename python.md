유용한 python library 함수들

### ord, chr

ord(문자): 유니코드 정수를 반환한다.

ord('a') == 97

chr(숫자): 숫자에 해당하는 문자 반환한다. ord 에 대응하는 함수이다.


### sort, sorted

sorted는 새로운 리스트를 반환한다. list 뿐만 아니라 dict의 경우도 사용할 수 있다.   
key라는 parameter에 lambda 함수를 넣어서 어떤 key로 정렬할지를 정할 수 있다.    
reverse=True 라는 parameter를 넣어서 역으로 정렬할 수도 있다.


sort의 경우는 list의 내장함수이다. 



### heapq

- heapq.heappush(heapname, value)
- heapq.heappop(heapname)
- heapq.nlargest(2, heapname): k개의 큰 원소를 리스트로 반환한다.

tuple을 넣는 경우 first element를 기준으로 비교한다. 만약 동일하다면 그 다음 element를 비교한다. 이 동작을 수정하려면 클래스를 새로 만들어서 `__lt__` 함수를 직접 구현한다.
https://www.geeksforgeeks.org/heapq-with-custom-predicate-in-python/




### bisect

`from bisect import bisect_left, bisect_right`

- bisect_left(list, data): 리스트에 데이터를 삽입할 가장 왼쪽 인덱스를 찾는 함수
- bisect_right(list, data): 리스트에 데이터를 삽입할 가장 오른쪽 인덱스를 찾는 함수

리스트 nums가 `[1 2 3 3 4 5]` 로 주어졌을 때, bisect_left(nums, 3)은 3을 넣을 수 있는 가장 왼쪽 index이므로 2가 되고 bisect_right(nums, 3)은 4가 된다.

### dict

d.items() 로 iterate하면 (key, value) 의 tuple을 iterate한다.   
d.values() 로 iterate하면 value만 iterate한다.

Get the key with the lowest value
```
>>> d = {320: 1, 321: 0, 322: 3}
>>> min(d, key=d.get)
321
```

dict에서 key 제거하기
`my_dict.pop('key', None)`: key가 없으면 None이 반환되고 있으면 `my_dict['key']`가 반환된다.   
`del my_dict['key']`: key가 없으면 KeyError가 raise된다.


sort하기   
```
sorted_tuples = sorted(a.items())
혹은
sorted_dict = OrderedDict(sorted(a.items()))
```

와 같이 할 수 있지만 느리다.

```
sorted_dict = {k: disordered[k] for k in sorted(disordered)}
```
이렇게 key 로만 하는 게 더 빠르다.


### copy

shallow copy: 실제로는 연결이 되어 있다. 메모리 주소만 복사한 것이고 같은 객체를 바라본다.    
immutable 객체의 경우는 shallow copy를 하든 deep copy를 하든 상관없다. 이 객체들은 값이 바뀌면 항상 참조가 바뀐다. 값 바뀔 때마다 객체가 새로 생기는 방식이다. 
그러면 a와 b가 같은 객체를 바라볼 때 a 값이 바뀌면 그 새로운 값을 위한 객체가 생성되고 a는 그 객체로 바라본다. b는 기존의 값을 바라본다.    

- `=`: 아예 리스트도 같은 객체를 바라본다.
  - a = [1,2,3,4], b = [5,6,7,8] 일 때 `a = b` 라고 한다면 a가 바라보는 객체는 b가 바라보는 리스트 객체와 동일해진다. 
  - 여기서 b = [0,0,0,0] 으로 다시 정의한다면 b가 바라보는 객체는 기존의 객체가 아니라 새로 생성된 리스트 객체가 된다. 따라서 a와 b는 다시 분리가 된다.
- `[:]`: 얕은 복사이다. 리스트 자체는 새로 만들지만 리스트 안의 객체는 같은 객체를 바라본다. 따라서 1d array의 경우는 깊은 복사처럼 동작하지만 2d array의 경우는 얕은 복사이다.    
- `copy.copy`: `[:]` 와 비슷하게 동작하는 것 같다.   
- `copy.deepcopy`: deep copy가 일어난다.



### compare

`==` 로 비교할 때, immutable은 값이 같은지를 확인한다.   
mutable은 reference를 확인한다. 값이 같아도 주소가 같아야한다.   



### Variable Scope

함수 안에 또 함수를 정의했을 때,

```py
class MyClass:
    def func1(self):
        a = 1

        def func2():
            # nonlocal a 라고 먼저 선언하면 사용할 수 있다.
            a += 1

        # func2를 호출하지 않으면 exception이 발생하지 않는다.
        # func2를 호출하면 UnboundLocalError: local variable 'a' referenced before assignment 에러가 발생한다.
        func2()  
        print(a)

mc = MyClass()
mc.func1()
```

func2에서 똑같은 이름의 변수명을 argument로 받게 되었을 때,

```py
class MyClass:
    def func1(self):
        a = 1

        def func2(a):
            a += 1
            print(f"func2: {a}")  # 101

        func2(100)
        print(f"func1: {a}")  # 1

mc = MyClass()
mc.func1()
```

func1의 parameter를 사용하려고 할 때

```py
class MyClass:
    def func1(self, a):

        def func2():
            # UnboundLocalError: local variable 'a' referenced before assignment
            a += 1
            print(f"func2: {a}")

        func2(a)
        print(f"func1: {a}")

mc = MyClass()
mc.func1(10)
```


func1의 paramter를 func2 argument로 넘겨줄 때

```py
class MyClass:
    def func1(self, a):

        def func2(a):
            a += 1
            print(f"func2: {a}")

        func2(a)
        print(f"func1: {a}")

mc = MyClass()
mc.func1(10)

"""
$ python3 tmp.py
func2: 11
func1: 10
"""
```

child function에서 수정된 건 반영되지 않는다.


nonlocal을 사용하는 경우는 child function에서 바꾼 것도 반영된다. 같은 주소를 참조하나보다.

```py
class MyClass:
    def func1(self, a):

        def func2():
            nonlocal a
            a += 1
            print(f"func2: {a}")

        func2()
        print(f"func1: {a}")

mc = MyClass()
mc.func1(10)

"""
func2: 11
func1: 11
"""
```
