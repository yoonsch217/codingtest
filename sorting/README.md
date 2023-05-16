# 개념

sorting의 기본은 우선 comparsion을 정의하는 것이다.   
영어 단어들이 있을 때 정렬 순서를 사전 순, 알파벳 수 순, 모음 수 순 등 여러 가지가 있다.   
`inputs.sort(key=lambda x: len(x))`

inversion이란, out of order인 pair를 말한다.   
`[3, 4, 6, 5, 2]` 의 리스트를 increasing order sorting 한다고 할 때 inversion은 `3,2` `4,2` `6,2` `5,2` `6,5` 총 다섯 개가 있다.   
즉, sorting이란 행동은 inversion 수를 0으로 만드는 작업과 같다고 말할 수 있다.   

stability라는 개념도 있다.   
valid한 sorting 결과가 여러 개일 수 있는데, 그 때 기존 input의 순서를 더 유지하는 결과가 더 stable하다고 말한다.


## Comparison Based Sort

### Selection Sort

맨 앞 element부터 차례대로 보면서 오른쪽으로 iterate 한 뒤에 가장 작은 값과 swap한다. 이렇게 n번 iterate하면 sorting된다.   
stable 하지 않은 sorting이다.

O(N^2) / O(1)



### Bubble Sort

맨 앞 두 개부터 차례대로 비교를 하면서 뒤에 element가 더 크면 swap한다.   
(0, 1) 비교하고 (1, 2) 비교하고 하면서 전체를 itearte한다.   
이 작업을 swap 없을 때까지 반복한다.
동일한 값끼리는 바꾸지 않기 때문에 stable한 sorting이다.   

O(N^2) / O(1)



### Insertion Sort

index 2 부터 오른쪽으로 iterate하면서 작업한다.   
각 element마다 왼쪽으로 iterate하면서 자기 위치에 멈춘다.   
이게 가능한 이유는, 우선 맨 처음 왼쪽 하나는 혼자니까 sorting 상태이고, 두 번째부터 차례대로 진행할 때는 그 index의 left array는 sorted된 것을 보장하기 때문이다.   
동일한 값과 바꿀 일이 없기 때문에 stable sort이다.    

inversion이 적을 때 유리한 방법이다. best case는 O(N) 일 것 같은데.   
또한 작은 array의 경우에 더 효과적이다(empirical observation). sorting function은 array size를 계산한 뒤에 특정 size 미만이면 insertion sort를 사용하기도 한다.(신기하다..)   

O(N^2) / O(1)




### Merge Sort




### Quick Sort

