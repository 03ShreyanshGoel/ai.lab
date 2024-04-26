def dfs(graph,initial,goal):
    stack=[initial]

    visited=set()
    while stack:
        node=stack.pop()
        print(node)
        if(node==goal):
            print("Goal is reached !")
            return 
        visited.add(node)

        for neighbour in graph[node]:
            if neighbour not in visited:
                stack.append(neighbour)

small_graph={'A': ['C', 'B'], 'B': ['C'], 'C': ['D'], 'D': []}

print("dfs traversal")
dfs(small_graph,'A','D')

