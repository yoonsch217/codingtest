### Course Schedule 2

https://leetcode.com/problems/course-schedule-ii/

문제: numCourses와 prerequisites가 주어진다. prerequisites은 어떤 수업을 듣기 위해서 다른 수업을 들어야할 때가 있는데 그 정보가 있다. 들어야하는 수업 순서를 반환하라. 불가능하면 `[]`를 반환하라.

Topological sort 문제이다. 먼저 defaultdict을 사용해서 adjacent list를 생성한다.    
그리고 각 노드마다 state를 둔다. WHITE는 방문한 적 없는, DFS를 시작할 노드이다. GREY는 방문한 적 있지만 아직 마지막까지 안 간 노드이다. BLACK은 그 노드에 대한 모든 작업을 마친 노드이다.   
모든 노드에 대해 iterate하면서 WHITE 상태이면 DFS를 한다. 그리고 DFS하면서 그 노드가 BLACK이 되면 topologically sorted array에 차례대로 추가를 한다.   
그 topologically sorted array를 뒤집은 게 정답이다.   
그리고 cycle이 있으면 불가능해진다. 따라서 DFS를 하다가 GREY를 다시 만난다면 cycle이 있는 것이므로 실패를 반환한다.
