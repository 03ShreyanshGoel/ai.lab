from queue import PriorityQueue

class Graph:
    def __init__(self):
        self.edges = {}
        self.weights = {}

    def neighbors(self, node):
        return self.edges.get(node, [])

    def get_cost(self, from_node, to_node):
        return self.weights.get((from_node, to_node), float('inf'))

def ucs(graph, initial, goal):
    visited = set()
    queue = PriorityQueue()
    queue.put((0, initial))
  
    while queue:
        cost, node = queue.get()
        if node in visited:
            continue
        
        visited.add(node)
        print(f"Visited Node: {node}, Cost: {cost}")
        
        if node == goal:
            print("Goal node reached!")
            return
        
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                total_cost = cost + graph.get_cost(node, neighbor)
                queue.put((total_cost, neighbor))

# Example usage with the graph
graph = Graph()
graph.edges = {'A': ['B', 'C'], 'B': ['D'], 'C': [], 'D': []}
graph.weights = {('A', 'B'): 1, ('A', 'C'): 2, ('B', 'D'): 3}
initial_node = 'A'
goal_node = 'D'

print(f"Starting UCS from {initial_node} to {goal_node}:")
ucs(graph, initial_node, goal_node)
