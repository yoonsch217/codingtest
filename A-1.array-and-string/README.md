# 개념

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

질문: StringBuilder 는 얼만큼의 시간 복잡도를 갖고 있나? O(N) 왜냐하면 append 자체가 amortized O(1) 이고 이걸 N번 반복하기 때문이다. 그리고 마지막에 stringfy 시키는 건 O(N)


Python: `''.join(str_list)`


## Sorting


sorting의 기본은 우선 comparsion을 정의하는 것이다.   
영어 단어들이 있을 때 정렬 순서를 사전 순, 알파벳 수 순, 모음 수 순 등 여러 가지가 있다.   
`inputs.sort(key=lambda x: len(x))`

inversion이란, out of order인 pair를 말한다.   
`[3, 4, 6, 5, 2]` 의 리스트를 increasing order sorting 한다고 할 때 inversion은 `3,2` `4,2` `6,2` `5,2` `6,5` 총 다섯 개가 있다.   
즉, sorting이란 행동은 inversion 수를 0으로 만드는 작업과 같다.   

stability라는 개념도 있다.   
valid한 sorting 결과가 여러 개일 수 있는데, 그 때 기존 input의 순서를 더 유지하는 결과가 더 stable하다고 말한다.

```
>> data = [('red', 1), ('blue', 1), ('red', 2), ('blue', 2)]
>> sorted(data, key=itemgetter(0))
[('blue', 1), ('blue', 2), ('red', 1), ('red', 2)]
# blue라는 두 원소에 대해 서로 동일한 sorting degree..?를 갖는데 이럴 땐 원래의 순서가 보장되도록 정렬된다.
```

### Selection Sort

- 시간 복잡도: O(N^2)
- 공간 복잡도: O(1)
- stable? No
- 동작 원리
  - 매 iteration 마다 최솟값을 찾아서 맨 앞부터 채워넣는다.
  - 맨 앞 element부터 차례대로 보면서 오른쪽으로 iterate 한 뒤에 가장 작은 값과 swap한다. 전체의 최솟값을 찾으면서 맨 앞에 넣고, 그 다음 최솟값 찾아서 그 다음에 넣는 방식이다. 이렇게 n번 iterate하면 sorting된다.   


<details>

```py
def selectionSort(self, nums):
    for i in range(len(nums)):
        _min = min(nums[i:])
        min_index = nums[i:].index(_min)
        nums[i + min_index] = nums[i]
        nums[i] = _min
    return nums
```

</details>


### Bubble Sort

- 시간 복잡도: O(N^2)
- 공간 복잡도: O(1)
- stable? Yes. 동일한 값끼리는 바꾸지 않는다.
- 동작 원리
  - 맨 앞 두 개부터 차례대로 비교를 하면서 앞에 element가 더 크면 swap한다.   
  - (0, 1) 비교하고 (1, 2) 비교하고 하면서 전체를 itearte한다.   
  - 물방울이 위로 뜨는 것처럼, 가장 큰 값을 뒤로 보내는 방식이다. 
  - 이 작업을 swap 없을 때까지 반복한다.
- selection sort 는 매 iteration 마다 전체를 탐색하고 왼쪽부터 최솟값이 채워진다. bubble sort 는 매 iteration 마다 인접 원소끼리만 비교하고 오른쪽부터 최댓값이 채워진다.


<details>

```py
def bubbleSort(self, nums):
    n = len(nums)
    for i in range(n):
        for j in range(0, n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                    
```

</details>



### Insertion Sort

- 시간 복잡도: O(N^2)
- 공간 복잡도: O(1)
- stable? Yes. 동일한 값과 바꿀 일이 없다.
- 동작 원리
  - index 2 부터 오른쪽으로 iterate하면서 작업한다.   
  - 각 element마다 왼쪽으로 iterate 하면서 자기보다 큰 값들을 오른쪽으로 shift 시킨다. 크지 않은 값을 만나면 그 자리에 넣고 멈춘다.   
  - 이게 가능한 이유는 왼쪽 subarray 는 정렬 상태를 보장하기 때문이다. 우선 맨 처음 왼쪽 하나는 혼자니까 sorting 상태이고, 두 번째부터도 sorting을 만족시킨다.
- inversion이 적을 때 유리한 방법이다. best case는 O(N) 일 것 같은데.   
- 작은 array의 경우에 더 효과적이다(empirical observation). 메모리를 순차적으로 접근하기 때문에 정렬 외의 오버헤드가 적다. (재귀 함수, 기타 로직들 등이 없음)
- sorting function은 array size를 계산한 뒤에 특정 size 미만이면 insertion sort를 사용하기도 한다.(오..) python 의 timsort 는 64 미만이면 insertion sort, 그 이상이면 merge sort 를 쓴다.  



<details>

```py
def insertionSort(self, nums): 
    for i in range(1, len(nums)): 
        key = nums[i]
        j = i-1
        while j >= 0 and key < nums[j] : 
                nums[j + 1] = nums[j] 
                j -= 1
        nums[j + 1] = key
```

</details>



### Heap Sort

- 시간 복잡도: O(N log N), 그러나 살제로는 배열을 순차적으로 접근하지 않기 때문에 bad cache locality 때문에 좀 느리다고 한다. 
- 공간 복잡도: O(1)
- stable? No.
- 동작 원리
  - selection sort에서는 매 iteration마다 minimum을 찾는데 minimum 찾는 게 O(N)의 시간복잡도를 갖는다. minimum 찾는 걸 빨리 해주는 heap을 사용한 알고리즘이 heap sort이다.   
  - unordered array를 bottom-up heapify 한다.
    - array를 arr[0]가 root인 binary tree로 볼 수 있다.
    - 목적은 arr[0]에 최댓값이 들어가게 하는 것이다.
    - arr[i] 입장에서 left child = arr[2 * i + 1],  right child = arr[2 * i + 2]
    - 맨 뒤의 node부터 앞으로 차례대로 오면서 각자의 child node가 자기보다 더 값이 크다면 swap을 한다. swap하고도 더 큰 child가 있다면 또 swap해준다. 이렇게 함으로써 각 subtree들도 max heap을 만족하게 된다.
  - arr[0]가 최댓값을 갖는다. arr[0]와 arr[len - 1] 를 swap한다. arr의 맨 마지막에는 최댓값이 들어가게 된다.
  - arr[0:len-1] 에 대해 동일하게 heapify한다. 이 때는 a[0]에 대해서만 자기가 있을 위치를 찾아서 넣어주면 된다. 왜나하면 이미 a[1]과 a[2]는 각각을 root로 하는 subtree의 max 값이기 때문이다.
  - heapify 할 때는 먼저 root에 대해 양 child와 비교한다. 그 중 right가 크다면 right와 swap한 뒤 그 right로 내려간 root 값에 대해 또 양 child와 비교한다. 이 과정을 swap이 일어나지 않을 때까지 반복한다.
  - arr[0:len-1] 의 max가 arr[0]에 오게 되면 그 값을 arr[len-2]와 swap 한 뒤에 arr[0:len-2]에 대해 동일하게 작업을 해준다.
- 오름차순 정렬할 때는 max heap이 더 편하다. 매 작업마다 최댓값을 맨 뒤로 보내면 되기 때문이다. 만약 minheap 을 쓴다면 매 작업마다 최솟값을 맨 앞으로 보내야하는데 그럼 root 값이 바뀌는 것이기 때문에 사용할 수 없고 별도의 메모리가 필요해지게 된다.


<details>

```py
def heap_sort(self, lst: List[int]) -> None:
    """
    Mutates elements in lst by utilizing the heap data structure
    """
    def max_heapify(heap_size, index):
        # index에 대해서 양 child와 비교한다. 현재 index에 대해서만 작업한다.
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

</details>





### Merge Sort

- 시간 복잡도: O(N log N) , 들어오는 데이터와 상관 없이 일정하다.
  - logN 번 merge를 해야한다. binary tree 로 나눈다고 했을 때 총 logN 의 층이 생기니까 각 층마다 merge가 이뤄진다고 생각하면 된다.
  - 각 층마다 merge 할 때는 결국 포인터가 한 번씩 다 훑기 때문에 O(N) 의 시간이 소요된다.
- 공간 복잡도: O(N) 
- stable? Yes
  - `L[i] <= R[j]` 이것처럼 동일할 때 왼쪽에 있던 원소가 먼저 들어가니까, 이 동점 처리 때문에 stability 가 보장된다. 
- 동작 원리: divide and conquer
  - 전체 array를 절반씩 쪼개서 하나만 남을 때까지 멈춘다. 하나만 있을 땐 정렬이 되어 있다.
  - 각각 정렬된 left subarray 와 right subarray 를 합친다. 각 subarray 마다 맨 앞에 포인터를 놓고 둘 중 작은 거를 뽑아서 새로 array를 만들면서 끝까지 iterate한다.
  - 이 과정을 전체 array를 만들 때까지 한다.


<details>

```py
# Merges two subarrays of arr[].
# First subarray is arr[l..m]. Second subarray is arr[m+1..r]
def merge(arr, l, m, r):
    # Get size of each subarray
    n1 = m - l + 1
    n2 = r - m
 
    # Create temp arrays
    L = [0] * (n1)
    R = [0] * (n2)
 
    # Copy data to temp arrays L[] and R[]
    for i in range(0, n1):
        L[i] = arr[l + i]
    for j in range(0, n2):
        R[j] = arr[m + 1 + j]
 
    # Merge the temp arrays back into arr[l..r]
    i = 0     # Initial index of first subarray
    j = 0     # Initial index of second subarray
    k = l     # Initial index of merged subarray
 
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
 
    # Copy the remaining elements of L[], if there are any
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
 
    # Copy the remaining elements of R[], if there are any
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1
 

# l is for left index and r is right index of the sub-array of arr to be sorted
def mergeSort(arr, l, r):
    if l < r:
        # Same as (l+r)//2, but avoids overflow for large l and h
        m = l+(r-l)//2
 
        # Sort first and second halves
        mergeSort(arr, l, m)
        mergeSort(arr, m+1, r)
        merge(arr, l, m, r)
```

</details>



### Quick Sort

- 시간 복잡도: Best O(N logN), worst O(N^2)
  - worst case는 pivot이 최소이거나 최대일 때이다. 따라서 배열이 이미 정렬/역정렬 되어 있다면 정렬이 n-1번 수행되어 최악의 경우이다. 
- 공간 복잡도: in-memory 이지만 recursive 로 인한 stack memory 사용량이 있다.
  - average: O(logN), worst: O(N)
- stable? No 
  - 병합 정렬은 인접한 값들을 비교하며 '차곡차곡' 합치지만, 퀵 정렬은 피벗을 기준으로 먼 거리의 원소를 점프하며 교환하기 때문에 원래의 상대적 순서를 보장할 수 없다. 
- 동작 원리 
  - divide and conquer
  - 처음에 전체 array 에 대해서 하나의 pivot 값을 정한다. (보통 rightmost)
  - 전체 array 를 iterate 하면서 그 값보다 작거나 같으면 왼쪽으로 보내고 크다면 그대로 둔다. left subarray 의 rightmost 포인터를 갖고 있어야한다.
  - 이 iteration 이 끝나면 left subarray 의 rightmost 바로 오른쪽에 pivot 값을 넣는다. 그렇게 되면 pivot 의 왼쪽은 자기보다 작거나 같고 오른쪽은 다 크다.
  - pivot 은 자기 위치를 찾아간 상태이다. 이제 pivot 을 기준으로 왼쪽 subarray 와 오른쪽 subarray 에 대해 동일하게 진행한다. 
- cache hit 측면에서 merge sort보다 효과적이다.

<details>

```py
def partition(array, low, high):
    # Choose the rightmost element as pivot
    pivot = array[high]
 
    # All elements that are on the left of i is less than or equal to pivot
    # i is the first index of the element that is greater than pivot.
    # i.e. After all the iterations, i will be the right next value of pivot.
    i = low

    # Traverse through all elements compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:
            # Swapping element at i with element at j
            array[i], array[j] = array[j], array[i]
            i += 1
 
    # Swap the pivot element with the greater element specified by i
    array[i], array[high] = array[high], array[i]
 
    # Return the position from where partition is done
    return i
 

def quicksort(array, low, high):
    if low < high:
        # The element at pi index is at the right position.
        # All the elements in the left subarray are less than or equal to pi. Right subarray vice versa.
        pi = partition(array, low, high)  # pi is on the right position
 
        # Recursive call on the left of pivot
        quicksort(array, low, pi - 1)
 
        # Recursive call on the right of pivot
        quicksort(array, pi + 1, high)
```

</details>


# 전략


### Sliding Window

- 개념 
  - window 크기가 정해져있는 경우에 nested loop 대신에 single loop를 사용함으로써 time complexity를 줄이는 것이 목적이다.
- 방법
  - 시작 지점에 left와 right라는 포인터 두 개를 놓는다.
  - 윈도우가 이동할 때 공통된 부분은 재사용하고, 새로 들어오는 데이터(Right)와 나가는 데이터(Left)만 처리하여 중복 계산을 제거한다.
- 활용
  - 예를 들어 주어진 int 리스트에서 연속된 k개의 합이 최대인 값을 찾으라는 문제가 있을 때, size k의 window를 만든 후 한 칸씩 이동하면서 이전 값을 빼고 새로운 값을 추가한다.    
  - 어떤 연속된 범위에서의 계산들이 필요할 때 사용하면 될 것 같다.   

### prefix sum

- 개념
  - 배열의 요소들을 차례대로 더한 값을 미리 저장해두어, 특정 구간의 합을 O(1) 만에 구하는 것이 목적이다. 
- 방법
  - 원본 배열을 순회하며 각 지점까지의 누적 합을 담은 prefix_sum 리스트를 생성한다. (인덱스 계산을 위해 맨 앞에 0을 넣는 Buffer를 둘 수도 있다.)
  - idx1부터 idx2까지의 구간 합은 pf[idx2 + 1] - pf[idx1] 공식을 사용하여 계산합니다.
  - 2d에서 prefix sum matrix를 구할 수도 있다. 
     - 이 때의 pf[i][j]의 값은 [i][j]보다 왼쪽에 있거나 위에 있는 모든 원소의 합이다. 
     - 그러면 `pf[i][j] = pf[i-1][j] + pf[i][j-1] - pf[i-1][j-1]`가 된다.
- 활용
  - 배열 내에서 구간 합 쿼리가 매우 빈번하게 들어올 때 사용한다.


### Two pointers

- 개념
  - 주로 정렬된 배열에서 두 개의 포인터를 조작하여 원하는 조건을 찾음으로써, O(N^2) 의 완전 탐색 문제를 O(N) 으로 최적화하는 것이 목적이다. 
- 방법
  - 배열의 양 끝(left, right) 혹은 같은 방향에서 시작하는 두 포인터를 둔다. 
  - 현재 두 포인터가 가리키는 값의 상태에 따라 포인터를 이동(Greedy하게 접근)시키며 탐색 범위를 좁혀나간다.
  - O(N^2) 이 나올 거 같으면 O(2N), O(3N) 등으로 할 수 있을지 생각해본다.
- 활용 
  - 정렬된 리스트에서 두 수의 합이 특정 값 S가 되는 쌍을 찾을 때 사용한다.
  - Container With Most Water 문제처럼 양쪽 끝에서 시작하여 면적을 비교하며 안쪽으로 좁혀 들어가는 최적화 문제에 유용하다. 

ex) https://leetcode.com/problems/container-with-most-water



### KMP Algorithm

https://leetcode.com/problems/longest-happy-prefix/description/
