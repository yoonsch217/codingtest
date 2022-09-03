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


### 2158. Amount of New Area Painted Each Day

https://leetcode.com/problems/amount-of-new-area-painted-each-day/

문제: 0-indexed 2D integer array paint가 주어진다. 각 element는 두 개의 원소를 갖는데 start position과 end position이다. paint array를 앞에서부터 iterate하면서 start position ~ end position 까지 색칠한다. 이전 작업에서 색칠된 부분은 더 색칠을 못한다. 각 iteration에서 색칠한 수를 구하라.

sweep line이라는 개념이 들어간다.   
https://leetcode.com/problems/amount-of-new-area-painted-each-day/discuss/1740812/Python-Complete-3-solutions-using-different-data-structures   
좀 어렵다.   
먼저 iteratre하면서 각각의 start position과 index를 tuple로 묶어서 리스트에 넣는다. end position과 index도 마찬가지로 넣는다. 각각이 start인지 end인지 boolean 등으로 표시를 해놓는다.    
그 다음에 position을 기준으로 sort를 한다.   
그러면 그 리스트를 앞에서부터 스캔하면 빠른 position부터 나올 것이다.   
그다음에는 전체 길이에 맞는 buffer array를 만든다.   
그러고는 position array를 iterate하면서 start가 나오면 그때부터 buffer의 해당 position에 해당 index를 넣는다. end가 나오면 해당 index에 대해서 그만 넣는다.   
이렇게 하면 나중에 buffer array를 살펴봤을 때 어떤 칸에 1, 5가 있다면 인덱스 1 작업과 인덱스 5 작업에 의해 색칠될 수 있던 공간이라는 뜻이다.   
인덱스 낮은 작업이 우선이므로 인덱스 1의 작업에 의한 색칠 부분으로 인식하면 된다.   





