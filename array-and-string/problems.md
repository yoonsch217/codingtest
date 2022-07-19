### 3. Longest Substring Without Repeating Characters

https://leetcode.com/problems/longest-substring-without-repeating-characters

문제: 문자열이 주어졌을 때 반복되는 글자가 없는 가장 긴 substring의 길이를 반환하라.

sliding window를 사용한다. substring의 left와 right를 정해주는 포인터 두 개를 저장한다.   
그리고 dictionary 하나를 만들어서 key는 나타났던 문자, value는 그 문자의 위치를 저장한다.   
string을 traverse하면서 현재 문자가 dictionary에 있으면서 그 문자의 위치가 cur_idx보다 크거나 같으면 left pointer를 `d[cur] + 1` 로 업데이트한다.   
그리고 `d[cur] = right` 로 업데이트 혹은 추가를 해주고 `right += 1` 로 포인터 위치를 옮긴다.   



### 1094. Car Pooling

https://leetcode.com/problems/car-pooling/

문제: capacity 라는 integer와 passengers라는 리스트가 주어진다. passengers 리스트는 `[승객수, 탑승 시각, 하차 시각]` 로 구성되어 있다. 전체 여정을 하면서 capacity를 넘는 순간이 있다면 false, 그렇지 않다면 true를 반환한다.

어떤 시간 순서열에서 i~j 까지 어떤 변화가 있다가 사라져야한다면 map[i] += event, map[j] -= event 식으로 사용한다.   
그러면 그 map을 iterate하면서 value 값을 더하거나 뺄 수 있고, iterate하다가 멈춘 지점의 상황을 알 수 있다.   
froms와 tos를 구해서 시간별로 더하고 뺀다.   
혹은 시각이 1~1000까지라는 제약이 있다면 bucket sort를 사용할 수도 있다. 