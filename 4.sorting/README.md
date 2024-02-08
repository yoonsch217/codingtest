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

맨 앞 두 개부터 차례대로 비교를 하면서 뒤에 element가 더 크면 swap한다.   
(0, 1) 비교하고 (1, 2) 비교하고 하면서 전체를 itearte한다.   
이 작업을 swap 없을 때까지 반복한다.
동일한 값끼리는 바꾸지 않기 때문에 stable한 sorting이다.   

O(N^2) / O(1)

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

index 2 부터 오른쪽으로 iterate하면서 작업한다.   
각 element마다 왼쪽으로 iterate하면서 자기 위치에 멈춘다.   
이게 가능한 이유는, 우선 맨 처음 왼쪽 하나는 혼자니까 sorting 상태이고, 두 번째부터 차례대로 진행할 때는 그 index의 left array는 sorted된 것을 보장하기 때문이다.   
동일한 값과 바꿀 일이 없기 때문에 stable sort이다.    

inversion이 적을 때 유리한 방법이다. best case는 O(N) 일 것 같은데.   
또한 작은 array의 경우에 더 효과적이다(empirical observation). sorting function은 array size를 계산한 뒤에 특정 size 미만이면 insertion sort를 사용하기도 한다.(신기하다..)   

O(N^2) / O(1)

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

divide and conquer   

- pointer들을 움직이면서 계속 절반씩 쪼갠다. 
- 하나만 남으면 sort된 상태이니까 하나 남을 때까지 쪼갠다.
- 쪼갠 거를 합치는데 각각의 subarray마다 포인터를 놓아서 비교해가며 채운다. O(N)


Time: O(N logN)

메모리가 많이 든다.

stable


<details>

```py
# Merges two subarrays of arr[].
# First subarray is arr[l..m]. Second subarray is arr[m+1..r]

def merge(arr, l, m, r):
    n1 = m - l + 1
    n2 = r - m
 
    # create temp arrays
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

divide and conquer   

- 어떤 array 안에서 pivot을 정한다. 
- pivot의 자기 위치를 찾아가며 그 값보다 작은 원소들은 left subarray로, 큰 원소들은 right subarray로 보낸다. 
- recursive하게 진행을 한다.   

특징   
- unstable sort
- 메모리를 많이 쓰지 않는다.    
- cache hit 측면에서 merge sort보다 효과적이다.   
- worst case는 pivot이 최소이거나 최대일 때이다. 따라서 배열이 이미 정렬/역정렬 되어 있다면 정렬이 n-1번 수행되어 최악의 경우이다.    
- Best O(N logN), worst O(N^2)

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
        pi = partition(array, low, high)
 
        # Recursive call on the left of pivot
        quicksort(array, low, pi - 1)
 
        # Recursive call on the right of pivot
        quicksort(array, pi + 1, high)
```

</details>


# 전략

보통은 array 문제로 나온다.   
그리고 built-in function을 주로 써서 이걸 직접 구현할 일은 잘 없을 것 같다.