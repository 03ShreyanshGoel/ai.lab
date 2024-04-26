from collections import deque

def bfs(graph,initial,goal):
    queue=deque([initial])
    visited=set()
    while queue:
        node=queue.popleft()
        print(node)
        if(node==goal):
            print("Goal reached !")
            return
        
        visited.add(node)
        for neighbour in graph[node]:
            if neighbour not in visited:
                queue.append(neighbour)


small_graph = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}

print("BFS Traversal:")
bfs(small_graph, 'A', 'D')

