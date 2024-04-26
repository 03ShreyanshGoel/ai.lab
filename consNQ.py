from constraint import Problem, AllDifferentConstraint

def n_queens_problem(n):
    # Initialize the constraint satisfaction problem
    problem = Problem()
    
    # Add variables for each column representing the queens' positions
    for i in range(n):
        problem.addVariable(f"Queen_{i + 1}", range(1, n + 1))
    
    # Add constraints to ensure no two queens attack each other
    problem.addConstraint(AllDifferentConstraint())
    for i in range(n):
        for j in range(i + 1, n):
            problem.addConstraint(lambda x, y, i=i, j=j: abs(x - y) != abs(i - j),
                                  (f"Queen_{i + 1}", f"Queen_{j + 1}"))
    
    # Get the solution
    solution = problem.getSolution()
    
    # Print the board
    if solution:
        for i in range(n):
            print(f"Row {i + 1}: Column {solution[f'Queen_{i + 1}']}")
    else:
        print(f"No solution found for {n}-Queens problem.")

# Call the function to solve the problem for N = 8
n_queens_problem(8)
