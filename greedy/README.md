

## 문제

https://leetcode.com/problems/boats-to-save-people/

문제: people 리스트가 주어지는데 각 값은 몸무게이다. 보트들에 사람을 최대 두 명 태울 수 있는데 `limit` 의 무게를 넘을 순 없다. 모든 사람을 태우기 위한 최소의 보트 수를 구하라.
limit 보다 무거운 사람은 없다.

풀이   
먼저 정렬을 한다. 그러고 two pointers로 오른쪽부터 하나, 왼쪽부터 하나를 놓고 이동한다.   
오른쪽부터 무거운 사람을 태우면서 left pointer 몸무게가 더 태울 수 있는 무게면 태우고 안 되면 만다.    
right pointer 이동할 때마다 보트를 사용하는 거니까 결과값을 1 늘린다.
