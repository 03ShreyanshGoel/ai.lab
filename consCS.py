from constraint import Problem, AllDifferentConstraint

def class_scheduling():
    # Define the time slots available for classes
    time_slots = ["Slot1", "Slot2", "Slot3", "Slot4"]
    
    # Define the classes and their corresponding professors
    classes = {
        "Class1": ["ProfA", "Room1"],
        "Class2": ["ProfB", "Room2"],
        "Class3": ["ProfC", "Room3"],
        "Class4": ["ProfD", "Room1"],
    }
    
    # Initialize the constraint satisfaction problem
    problem = Problem()
    
    # Add variables for each class with possible time slots
    for class_name, class_info in classes.items():
        problem.addVariable(class_name, time_slots)
    
    # Add constraints
    for class_name, class_info in classes.items():
        professor, room = class_info
        problem.addConstraint(AllDifferentConstraint(), [c for c in classes if classes[c][0] == professor])
        problem.addConstraint(AllDifferentConstraint(), [c for c in classes if classes[c][1] == room])
    
    # Get the solution
    solution = problem.getSolution()
    
    # Print the schedule
    if solution:
        for class_name, time_slot in solution.items():
            print(f"{class_name} scheduled at {time_slot}")
    else:
        print("No valid schedule found.")

# Call the function to solve the problem
class_scheduling()
