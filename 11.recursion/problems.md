### 52. N-Queens II

https://leetcode.com/problems/n-queens-ii/

문제: nxn 체스판에 n 개의 퀸을 놓아야하는데 서로 공격을 못 하게 하는 unique 배치의 수를 구하라.

<details><summary>Approach 1</summary>

same column, same row, diagonal, anti-diagonal 을 피해서 배치해야한다.   

- 같은 row를 피하는 방법으로는 각 작업마다 하나의 row씩 늘려서 배치하는 것이 있다.   
- 같은 column을 피하는 방법으로는 column set을 만들어서 set 안에 있는지 확인할 수 있다.   
- diagonal과 anti-diagonal이 조금 독특한데 diagonal position에 있으려면 `(compare_row - compare_col) == (cur_row - cur_col)` 이 되어야 하고, 
anti-diagonal position에 있으려면 `(compare_row + compare_col) == (cur_row + cur_col)` 가 되어야한다.   
따라서 row-col 을 보관하는 diagonal set과 row+col 을 보관하는 anti-diagonal set을 갖고 비교하면 된다.   

recursion으로 row를 늘려가면서 invalid한 순간 멈추고 backtracking하여 다음 candidate를 검증하면 된다.   




```python
    def totalNQueens(self, n: int) -> int:
        def get_valid_positions(row, diag_set, anti_diag_set, col_set):
            if row == n:  # base case 잊지 말기
                return 1
            res = 0
            for i in range(n):
                cur_diag = row - i
                cur_anti_diag = row + i
                if i not in col_set and cur_diag not in diag_set and cur_anti_diag not in anti_diag_set:
                    col_set.add(i)
                    diag_set.add(cur_diag)
                    anti_diag_set.add(cur_anti_diag)
                    res += get_valid_positions(row+1, diag_set, anti_diag_set, col_set)
                    col_set.remove(i)
                    diag_set.remove(cur_diag)
                    anti_diag_set.remove(cur_anti_diag)
            
            return res
        
        res = 0
        for i in range(n):
            diag_set = set()  # stores row-col values
            anti_diag_set = set()  # stores row+col values
            col_set = set()

            diag_set.add(-i)
            anti_diag_set.add(i)
            col_set.add(i)
            res += get_valid_positions(1, diag_set, anti_diag_set, col_set)
        return res

```

Complexity:   
- O(N!) / O(N) set 비교하는 건 O(1)이니까 처음에 N개, 그 다음에 N-1, ... 해서 N!이다.    


</details>











### 489. Robot Room Cleaner

https://leetcode.com/problems/robot-room-cleaner/

문제: robot이 2차원 matrix 형태인 방을 청소한다. robot은 네 방향을 바라보고 한 칸씩 아동한다. 
robot 객체에는 move, turnRight, turnLeft, clean 네 가지의 함수가 있다. move는 현재 바라보는 방향으로 한 칸을 이동할 수 있으면 이동하고 True를 반환하고 못 가면 가만히 있으면서 False를 반환한다.
이 네 가지 함수를 사용하여 주어진 2차원 matrix의 방을 다 청소하는 함수를 짜라. 방 matrix는 주어지지 않는다.

<details><summary>Approach 1</summary>

방문한 곳은 다시 방문하지 않는 것이 좋다. 따라서 visited set을 만들어서 들고 다닌다. 
4 방향 다 살펴봤을 때 더이상 갈 곳이 없다면 처음의 위치로 backtracking을 한다. 이렇게 함으로써 맨 처음 기준으로 네 방향을 다 탐색할 수가 있다.

DFS 랑 비슷하다. DFS에서는 child 두 개 중 하나를 골라서 끝까지 갔다가 backtracking해서 나머지 하나로 또 끝까지 간다. robot clean의 경우는 child가 네 개인 상황으로 생각하면 된다. 
한 방향을 끝까지 탐색해서 더 갈 곳이 없으면 backtrack해서 원래 자리로 돌아온 뒤 다른 child로 가야한다.
    
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


<details><summary>Approach 1</summary>

`x^n = x^(n//2) * x^(n//2) * x^(n%2)` => base case는 exponent가 0 혹은 1일 때이다.

```py
def myPow(self, x: float, n: int) -> float:
    @lru_cache(maxsize=None)
    def get_pow(base, exponent):
        if exponent == 0:
            return 1
        elif exponent % 2 == 0:
            return get_pow(base * base, exponent // 2)
        else:
            return base * get_pow(base * base, (exponent - 1) // 2)

    f = get_pow(x, abs(n))
    
    return float(f) if n >= 0 else 1/f
```

</details>







### 93. Restore IP Addresses

https://leetcode.com/problems/restore-ip-addresses/description/

문제: 길이가 1에서 15까지인 문자열이 주어졌고 각 문자는 digit이다. 이 문자열에 dot 세 개를 넣어서 만들 수 있는 valid ip address의 리스트를 반환하라. 
각 숫자는 0 ~ 255 까지의 값이어야하고 cannot have leading zeros의 조건을 만족해야한다.

<details><summary>Approach 1</summary>

iterative한 건, is_valid(start_idx, end_idx)를 만들어 놓고 3중 for 문을 통해 각 위치에 dot을 넣는 것이다. 그래서 만들어진 4개의 part가 다 valid하면 정답에 추가한다.    

```python
def restoreIpAddresses(self, s: str) -> List[str]:
    n = len(s)
    if n > 12 or n < 4:
        return []

    @lru_cache(maxsize=None)
    def get_valid_value_or_none(start_idx, end_idx):
        if end_idx - start_idx >= 3:
            return None
        if start_idx != end_idx and s[start_idx] == '0':
            return None
        value = 0
        right = end_idx
        while start_idx <= right:
            value += int(s[right]) * pow(10, end_idx - right)
            right -= 1
        if 0 <= value <= 255:
            return s[start_idx: end_idx+1]
        return None

    ans = []
    for i in range(3):
        for j in range(i+1, n-1):
            if j >= i+4:
                break
            for k in range(j+1, n-1):
                if k >= j+4:
                    break
                first = get_valid_value_or_none(0, i)
                second = get_valid_value_or_none(i+1, j)
                third = get_valid_value_or_none(j+1, k)
                fourth = get_valid_value_or_none(k+1, n-1)
                if any(res is None for res in [first, second, third, fourth]):
                    continue
                ans.append('.'.join([first, second, third, fourth]))

    return ans
```

</details>
    

<details><summary>Approach 2</summary>
    
backtracking하는 건, dots 위치 리스트를 갖고 다니면서 backtrack 시작하기 전에 dots.append(cur_dot_idx)하고 끝나면 dots.pop() 을 한다.   
각 dot마다 iterate할 때는 세 번만 iterate하면 된다.   


```python
    def restoreIpAddresses(self, s: str) -> List[str]:
        n = len(s)
        ans = []
        @lru_cache(maxsize=None)
        def get_valid_seq_or_none(start_idx, end_idx):
            if end_idx - start_idx >= 3 or end_idx >= n:
                return None
            if start_idx != end_idx and s[start_idx] == '0':
                return None
            value = 0
            right = end_idx
            while start_idx <= right:
                value += int(s[right]) * pow(10, end_idx - right)
                right -= 1
            if 0 <= value <= 255:
                return s[start_idx: end_idx+1]
            return None
        
        tmp_list = []
        def backtrack(start_idx, remained_dots):
            if start_idx >= n:
                return
            if remained_dots == 0:
                valid_seq = get_valid_seq_or_none(start_idx, n-1)
                if valid_seq:
                    tmp_list.append(valid_seq)
                    ans.append('.'.join(tmp_list))
                    tmp_list.pop()

            # start idx is the very next idx of the latest dot
            # Verify if valid and put dot
            for i in range(3):
                valid_seq = get_valid_seq_or_none(start_idx, start_idx + i)
                if valid_seq:
                    tmp_list.append(valid_seq)
                    backtrack(start_idx + i + 1, remained_dots - 1)
                    tmp_list.pop()

        
        backtrack(0, 3)
        return ans
```


내 솔루션. 어떻게 풀긴 풀었네. 80%


```py
    def restoreIpAddresses(self, s: str) -> List[str]:
        if len(s) < 4:
            return []
        
        def is_valid(target):
            if len(target) == 1:
                return True
            if len(target) == 2 and target[0] != '0':
                return True
            if len(target) == 3 and target[0] != '0' and int(target) < 256:
                return True
            return False
        
        def get_ip(raw_str, dots):
            tmp = []
            prev = 0
            for dot in dots:
                tmp.append(raw_str[prev:dot+1])
                prev = dot + 1
            tmp.append(raw_str[prev:])
            return '.'.join(tmp)


        def get_possible_ips(cur_idx, dots, res):
            # if last_dot_idx is i, it means that there's a dot just after i-th character
            if len(dots) > 3:
                return
            if len(dots) == 0:  
                # initial condition
                last_dot_idx = -1
            else:
                last_dot_idx = dots[-1]

            if cur_idx == len(s):
                # when reached the right end, add to the result if the last section is valid
                if len(dots) == 3:
                    if is_valid(s[last_dot_idx+1:]):
                        res.append(get_ip(s, dots))
                return

            if cur_idx - last_dot_idx > 3:
                return
            if is_valid(s[last_dot_idx+1:cur_idx+1]):
                get_possible_ips(cur_idx+1, dots, res)
                dots.append(cur_idx)
                get_possible_ips(cur_idx+1, dots, res)
                dots.pop()
        
        res = []
        get_possible_ips(0, [], res)
        return res
```

</details>






### 494. Target Sum

https://leetcode.com/problems/target-sum/description/

문제: You are given an integer array nums and an integer target. 
You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers. 
For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1". 
Return the number of different expressions that you can build, which evaluates to target. 

<details><summary>Approach 1</summary>

전형적인 backtracking 문제이다.

```py
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        @lru_cache(maxsize=None)
        def backtrack(idx, current_sum):
            if idx == len(nums):
                if current_sum == target:
                    return 1
                return 0
            
            plus_res = backtrack(idx+1, current_sum + nums[idx])  # 여기서 recursive하게 들어갈 때랑 
            minus_res = backtrack(idx+1, current_sum - nums[idx])  # 여기서 들어갈 때랑 겹칠 수가 있다. 그 부분을 lru_cache로 최적화한다.
            
            return plus_res + minus_res
            
        
        res = backtrack(0, 0)
        return res
```

</details>





### Target Sum (interview)

문제: 1, 2, ..., 9의 숫자가 순서대로 있고 그 사이에 +, -를 넣든가 아무것도 안 넣을 수 있다. 그렇게 만들어진 수식이 100이 되도록 하는 수식을 모두 구하라.

<details><summary>Apporach 1</summary>

```py
def find_expressions(target):
    def backtrack(start, expression, current_sum):
        if start == 9:
            if current_sum == target:
                expressions.append(expression)
            return

        # Try adding the next number
        backtrack(start + 1, expression + '+' + str(start + 1), current_sum + (start + 1))
        
        # Try subtracting the next number
        backtrack(start + 1, expression + '-' + str(start + 1), current_sum - (start + 1))
        
        # Try concatenating the next number
        new_number = int(str(start) + str(start + 1))
        backtrack(start + 2, expression + str(new_number), current_sum + new_number)

    expressions = []
    backtrack(1, '1', 1)
    return expressions

target_number = 100  # Change this to your target number
result = find_expressions(target_number)
for expression in result:
    print(expression)
```

</details>


