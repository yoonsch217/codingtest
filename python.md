유용한 python library 함수들

### ord, chr

ord(문자): 유니코드 정수를 반환한다.

ord('a') == 97

chr(숫자): 숫자에 해당하는 문자 반환한다. ord 에 대응하는 함수이다.


### heapq

- heapq.heappush(heapname, value)
- heapq.heappop(heapname)
- heapq.nlargest(2, heapname): k개의 큰 원소를 리스트로 반환한다.

tuple을 넣는 경우 first element를 기준으로 비교한다. 만약 동일하다면 그 다음 element를 비교한다. 이 동작을 수정하려면 클래스를 새로 만들어서 `__lt__` 함수를 직접 구현한다.
https://www.geeksforgeeks.org/heapq-with-custom-predicate-in-python/




### bisect


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

