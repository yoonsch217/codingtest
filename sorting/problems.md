### 75. Sort Colors

https://leetcode.com/problems/sort-colors

문제: nums 라는 integer list가 있고 0, 1, 2의 숫자가 있다. 0, 1, 2 순서대로 숫자들이 모이도록 정렬해라. in place.

selection sort를 사용하면 된다.   
맨 앞 element 부터 차례대로, 오른쪽으로 iterate하면서 최솟값과 swap을 한다.

아니면 각각 count를 세서 앞에서부터 채워도 된다. 

근데 solution에 신박한 one pass algorithm이 있다.   
Dutch National Flag 이라는 solution인데 포인터 세 개를 이용하는 방법이다.   
올바른 결과에서 0은 왼쪽부터, 2는 오른쪽부터 채워지면 되고 1은 나머지에 있으면 된다.   
그러면 p0을 제일 왼쪽, p2를 제일 오른쪽으로 둔다.   
그러고 cur 라는 포인터를 왼쪽부터 iterate하면서 0이면 p0과 swap하고 p0 한 칸 올리기, 2면 p2와 swap 후 p2 한 칸 내리기, 1이면 skip 하고 cur 올리면 된다.

