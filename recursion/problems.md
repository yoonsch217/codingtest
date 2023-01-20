### 52. N-Queens II

https://leetcode.com/problems/n-queens-ii/

문제: nxn 체스판에 n 개의 퀸을 놓아야하는데 서로 공격을 못 하게 하는 unique 배치의 수를 구하라.

same column, same row, diagonal, anti-diagonal 을 피해서 배치해야한다.   
같은 row를 피하는 방법으로는 각 작업마다 하나의 row씩 늘려서 배치하는 것이 있다.   
같은 column을 피하는 방법으로는 column set을 만들어서 set 안에 있는지 확인할 수 있다.   
diagonal과 anti-diagonal이 조금 독특한데 diagonal position에 있으려면 `(compare_row - compare_col) == (cur_row - cur_col)` 이 되어야 하고, anti-diagonal position에 있으려면 `(compare_row + compare_col) == (cur_row + cur_col)` 가 되어야한다.   
따라서 row-col 을 보관하는 diagonal set과 row+col 을 보관하는 anti-diagonal set을 갖고 비교하면 된다.   
recursion으로 row를 늘려가면서 invalid한 순간 멈추고 backtracking하여 다음 candidate를 검증하면 된다.   

혹은 내 방식으로는, occupied 리스트를 만들어서 각 순간마다 is_valid를 판단한다.   
각 recursion마다 row를 증가시키니까 row validity는 건너뛰고 column validity는 `cur_col in occupied` 로 비교한다.   
diagonal or anti-diagonal의 경우는 `abs(cur_row - compare_row) == abs(cur_col - compare_col)` 으로 할 수 있다.   
recursion이 끝날 때마다 occupied.pop() 을 해줘서 불필요한 copy를 막음으로써 시간 복잡도를 줄인다.   

<details>
    
```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        self.cnt = 0
        self.occupied = []

        def is_valid(cur_row, cur_col):
            for i in range(len(self.occupied)):
                compare_row = i
                compare_col = self.occupied[i]
                if cur_col == compare_col:
                    return False
                if abs(cur_row - compare_row) == abs(cur_col - compare_col):
                    return False
            return True

        def helper(row):
            if row == n:
                self.cnt += 1
                return
            for i in range(0, n):
                if is_valid(row, i):
                    self.occupied.append(i)
                    helper(row+1)
                    self.occupied.pop()

        helper(0)
        return self.cnt
```

</details>

Complexity:   
solution 대로 하면 O(N!)/O(N)이다. set 비교하는 건 O(1)이니까 처음에 N개, 그 다음에 N-1, ... 해서 N!이다.    
확실히 빠르네 솔루션이.


### 489. Robot Room Cleaner

https://leetcode.com/problems/robot-room-cleaner/

문제: robot이 2차원 matrix 형태인 방을 청소한다. robot은 네 방향을 바라보고 한 칸씩 아동한다. 
robot 객체에는 move, turnRight, turnLeft, clean 네 가지의 함수가 있다. move는 현재 바라보는 방향으로 한 칸을 이동할 수 있으면 이동하고 True를 반환하고 못 가면 가만히 있으면서 False를 반환한다.
이 네 가지 함수를 사용하여 주어진 2차원 matrix의 방을 다 청소하는 함수를 짜라. 방 matrix는 주어지지 않는다.

방문한 곳은 다시 방문하지 않는 것이 좋다. 따라서 visited set을 만들어서 들고 다닌다. 네 방향 다 살펴봤을 때 더이상 갈 곳이 없다면 처음의 위치로 backtracking을 한다. 이렇게 함으로써 맨 처음 기준으로 네 방향을 다 탐색할 수가 있다.

<details>
    
```python
class Solution:
    def cleanRoom(self, robot):
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        visited = set()
        
        def move_back():
            robot.turnRight()
            robot.turnRight()
            robot.move()
            robot.turnRight()
            robot.turnRight()
        
        def helper(pos, direction):  # directions: 0-up, 1-right, 2-down, 3-left
            if pos in visited:
                return
            robot.clean()
            visited.add(pos)
            for i in range(4):
                next_direction = (direction + i) % 4
                dx, dy = directions[next_direction]
                next_pos = (pos[0]+dx, pos[1]+dy)
                if next_pos not in visited and robot.move():
                    helper(next_pos, next_direction)
                    move_back()
                robot.turnRight()
        
        helper((0, 0), 0)
```

</details>

### 50. Pow(x, n)

https://leetcode.com/problems/powx-n/

문제: x의 n제곱을 구하라.

n의 최대 범위는 2^32-1 이다. 그러면 2진법으로 생각했을 때 32개의 숫자로 표현할 수 있는 거다.   
n은 1, 2, 4, 8, ... 로 표현할 수 있다. 각각은 0으로도 쓸 수 있고. 그러면 pow(2, n)은 `pow(2, 0) * pow(2, 1) * pow(2, 2) * pow(2, 4) * pow(2, 8) * ...` 로 표현할 수 있다. (각각은 1로도 쓸 수 있고)
pow(x, 1), pow(x, 2), pow(x, 4), pow(x, 8) 이렇게 2의 거듭제곱으로 지수를 올리면서 n에 제일 가깝게 올라간다.   
그리고 그 각각의 값들을 dict에 저장한다.   
최대한 올라간 지수를 cur_pow라고 하면 `n-cur_pow` 에 대한 문제라고 생각할 수 있다.    
이렇게 특정 지수 target_power에 대해 2의 거듭제곱으로 증가하면서 넘기 직전까지 가는 recursive 함수를 구한 뒤 그걸 사용한다.   
base case는 target_power가 0 혹은 1일 때이다.   







