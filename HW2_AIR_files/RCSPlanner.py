import numpy as np

class RCSPlanner(object):    
    def __init__(self, planning_env):
        self.planning_env = planning_env

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = [] 

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # TODO: Task 4

        return np.array(plan)

    def reconstruct_path(self, node):
        '''
        Reconstruct the path from the goal to the start using parent pointers.
        # YOU DON'T HAVE TO USE THIS FUNCTION!!!
        '''
        path = []
        while node:
            path.append(node.state)  # Append the state
            node = node.parent  # Move to the parent
        path.reverse()
        print(path)
        return np.array(path)
    
    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        DO NOT MODIFY THIS FUNCTION!!!
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes
