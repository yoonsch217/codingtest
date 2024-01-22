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



### Heap Sort

selection sort에서는 매 iteration마다 minimum을 찾는데 minimum 찾는 게 O(N)의 시간복잡도를 갖는다.   
minimum 찾는 걸 빨리 해주는 heap을 사용한 알고리즘이 heap sort이다.   

min heap 도 사용할 수 있지만 max heap이 더 편하다. 

1. unordered array를 bottom-up heapify 한다.
   - array를 arr[0]가 root인 binary tree로 볼 수 있다.
   - 목적은 arr[0]에 최댓값이 들어가게 하는 것이다.
   - arr[i]의 left child는 arr[2 * i + 1] 이고 right child는 arr[2 * i + 2] 가 된다.
   - 맨 뒤의 node부터 앞으로 차례대로 오면서 각자의 child node가 자기보다 더 값이 크다면 swap을 한다. swap하고도 더 큰 child가 있다면 또 swap해준다. 이렇게 함으로써 각 subtree들도 max heap을 만족하게 된다.
2. arr[0]가 최댓값을 갖는다. arr[0]와 arr[len - 1] 를 swap한다. arr의 맨 마지막에는 최댓값이 들어가게 된다.
3. arr[0:len-1] 에 대해 동일하게 heapify한다. 이 때는 a[0]에 대해서만 위치를 찾아주면 된다. 왜나하면 이미 a[1]과 a[2]는 각각을 root로 하는 subtree의 max 값이기 때문이다.
   - heapify 할 때는 먼저 root에 대해 양 child와 비교한다. 그 중 right가 크다면 right와 swap한 뒤 그 right로 내려간 root 값에 대해 또 양 child와 비교한다. 이 과정을 swap이 일어나지 않을 때까지 반복한다.
4. arr[0:len-1] 의 max가 arr[0]에 오게 되면 그 값을 arr[len-2]와 swap 한 뒤에 arr[0:len-2]에 대해 동일하게 작업을 해준다.

대부분의 다른 comparison based sort보다 빠르다.   
하지만 stable한 sort가 아니다.    
그리고 실제로는 bad cache locality 때문에 O(N log N) 보다 느린 것으로 알려졌다. locations in heaps 를 기반으로 swap을 하는데 이는 무작위로 정렬된 곳에서 index를 접근하기 위해 많은 read operation이 필요하여서 cache miss가 발생한다.   


```py
def heap_sort(self, lst: List[int]) -> None:
    """
    Mutates elements in lst by utilizing the heap data structure
    """
    def max_heapify(heap_size, index):
        # index에 대해서 양 child와 비교한다.
        left, right = 2 * index + 1, 2 * index + 2
        largest = index
        if left < heap_size and lst[left] > lst[largest]:
            largest = left
        if right < heap_size and lst[right] > lst[largest]:
            largest = right
        if largest != index:
            # 양 child가 없거나 양 child보다 내가 더 크면 거기서 작업은 끝나게 된다.
            # 작업이 안 끝난다면 내려간 곳에서 또 양 child에 대한 작업을 하게 되는 것이다.
            lst[index], lst[largest] = lst[largest], lst[index]
            max_heapify(heap_size, largest)

    # heapify original lst
    for i in range(len(lst) // 2 - 1, -1, -1):
        max_heapify(len(lst), i)

    # use heap to sort elements
    for i in range(len(lst) - 1, 0, -1):
        # swap last element with first element
        lst[i], lst[0] = lst[0], lst[i]
        # note that we reduce the heap size by 1 every iteration
        max_heapify(i, 0)
```





### Merge Sort




### Quick Sort

