### 49. Group Anagrams

https://leetcode.com/problems/group-anagrams

문제: 문자열 리스트가 주어졌을 때 같은 anagram끼리 묶은 리스트를 반환하라. 리스트 순서는 상관 없다.

두 문자열이 anagram인지를 확인하기 위해서는 sorting 결과를 보는 방법도 있고 각 문자의 출현 횟수를 비교하는 방법도 있다.   
지금 문제의 경우는 알파벳 소문자만 사용하기 때문에 sorting보다는 출현 횟수를 보는 게 더 빠르다.

defaultdict(list)를 만든 후 각 문자열마다 사용 횟수를 저장할 [0] * 26의 리스트를 만든다.   
ord() 함수로 count를 증가시킨 다음에 마지막에 counts list를 hashable key로 만들어야한다.   
이 때, 나는 map(str, counts) 한 뒤에 '.'.join(counts)를 했는데 이러면 너무 느리다.   
대신 tuple(counts)를 하면 빠른 시간에 hashable key 생성이 된다.




