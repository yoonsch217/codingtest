### 739. Daily Temperatures

https://leetcode.com/problems/daily-temperatures

문제: temperatures라는 리스트가 있는데 하루 간격의 기온이 저장되어 있다. 각 날짜에서 더 따뜻한 날이 올 때까지 기다려야하는 일수를 저장한 리스트를 반환하라. 더 따뜻한 날이 이후에 없다면 0을 저장하면 된다.

decreasing monotonic stack을 사용한다.   
리스트를 iterate하면서 stack에는 index만 저장을 하게 되는데 현재 기온이 stack의 마지막보다 높다면 현재 기온보다 높은 날이 나올 때까지 pop을 한다.   
pop하는 원소는 그 index와 현재 index의 차이만큼 기다리면 현재 기온을 만나게 되는 것이므로 answer 리스트에는 그 index 차이를 저장한다.   
각 원소에 대해 한 번씩만 작업을 하게 되므로 O(N) 시간이 걸리게 되고 stack을 위한 O(N) 공간이 필요하게 된다.   

혹은 리스트를 뒤에서부터 iterate하면서 현재 날짜의 기온보다 높은 기온이 나오는 날을 찾는 방법도 있다.   
지금까지의 가장 높은 기온을 저장하는 hottest variable을 두고 현재 기온이 hottest보다 높다면 hottest를 업데이트하고 continue한다.   
이렇게 하는 이유는 그런 경우 더 따뜻한 날이 나올 수 없으므로 추가 작업이 필요 없기 때문이다.   
answer list를 만들어 놓고 뒤에서부터 원소를 하나씩 보는데, i번째 날에 i+1의 온도를 확인한다.   
i+1의 온도가 더 낮다면 i+1+answer[i+1] 위치로 가서 또 비교한다.   
더 높은 온도가 나올 때까지 반복을 하는데 이렇게 하면 각 원소마다 두 번씩만 작업을 하게 된다.(backward iterate할 때 한 번, jump하면서 날 찾을 때 한 번)   
따라서 O(N) time에 O(1) space 답을 해결할 수 있다.   



### 42. Trapping Rain Water

https://leetcode.com/problems/trapping-rain-water/

