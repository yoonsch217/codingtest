## 개념

### ArrayList & Resizable Arrays

몇몇 언어에서 array(혹은 list)는 자동으로 사이즈가 조정된다.   
자바 같은 다른 언어에서는 array의 길이는 고정이 되어 있고 처음 생성할 때 사이즈를 정의한다.   

ArrayList와 같은 자료구조는 array가 가득 찼을 때 사이즈를 두 배로 늘려주는데 O(n)의 시간이 필요하다.   
하지만 가끔 일어나기 때문에 amortize 되어 insertion time은 평균 O(1)으로 말할 수 있다.   
하지만 array full일 때 수행되는 특정 insertion은 O(n)의 시간이 걸릴 수 있다.   


### StringBuilder

길이 x의 string n개를 하나하나 iterate하면서 합친다고 하면 O(x + 2x + ... + nx) = O(xn^2)의 시간이 걸린다.   
이를 효율적으로 하기 위해 StringBuilder를 사용한다.   
StringBuilder는 resizable array를 만들어서 그 array에 계속 append를 한다.   

Python: `''.join(str_list)`


## 전략


### Sliding Window

nested loop 대신에 single loop를 사용함으로써 time complexity를 줄이는 것이 목적이다.   
window 크기가 정해져있는 경우에 사용할 수 있다.   
시작 지점에 left와 right라는 포인터 두 개를 놓는다.   

예를 들어 주어진 int 리스트에서 연속된 k개의 합이 최대인 값을 찾으라는 문제가 있을 때, size k의 window를 만든 후 한 칸씩 이동하면서 이전 값을 빼고 새로운 값을 추가한다.   
이렇게 하면 매 작업마다 k번 더하는 연산을 할 필요 없이 두 번만 연산을 하면 된다.   
어떤 연속된 범위에서의 계산들이 필요할 때 사용하면 될 것 같다.   

### prefix sum

list를 traverse 하면서 지금까지의 합을 저장하는 prefix_sum_list를 생성한다.   
그러면 idx1~idx2 범위의 값의 합을 구할 때 `pf[idx2] - pf[idx1-1]` 로 구할 수 있으므로 O(1)의 시간이 걸린다.   
이럴 때 pf 리스트는 앞과 뒤에 buffer로 0을 추가하는 것이 좋다.    

2d에서 prefix sum matrix를 구할 수도 있다.   
이 때의 pf[i][j]의 값은 [i][j]보다 왼쪽에 있거나 위에 있는 모든 원소의 합이다.   
그러면 `pf[i][j] = pf[i-1][j] + pf[i][j-1] - pf[i-1][j-1]`가 된다.   



### Two pointers

Brute force하게 2 depth iteration을 해야해서 O(N^2) 시간이 걸리는 상황에서 최적화를 고민할 때 two poitner를 생각해보자.   
양 쪽에 pointer를 두고 greedy한 로직을 생각하면 one pass로 할 수도 있다.    
O(N^2) 이 나올 거 같으면 O(2N), O(3N) 등으로 할 수 있을지 생각해본다.

ex) https://leetcode.com/problems/container-with-most-water


