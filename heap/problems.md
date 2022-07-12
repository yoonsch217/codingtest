
### 378. Kth Smallest Element in a Sorted Matrix

https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix

문제: 이차원 매트릭스의 각 행과 열이 정렬이 되어 있다. 전체 숫자 중 작은 것부터 순서대로 셌을 때 k 번째인 값을 구하라.

external sort처럼 생각을 해보면, 각 행이 정렬되어 있기 때문에 각 행의 맨 앞끼리만 비교하면서 k번째인 값을 구하면 된다.   
리스트가 두 개라면 포인터를 하나씩 두면서 할 수 있지만 N개의 포인터를 두는 건 쉽지 않다.   
이럴 때는 힙을 사용하면 쉽게 해결할 수 있다.    
힙에 각 `(row의 head 값, row, col)` 를 넣고 k 번 iterate하면 된다.   


