import numpy as np
from RRTTree import RRTTree
import time


class RRTPlanner(object):
    def __init__(self, planning_env, ext_mode, goal_prob):
        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

        # our added variables
        self.eta = 15.0  # step size for E2
        self.goal_threshold = 1e-5  # thr to considering "close enough"
        self.max_iter = 10000  # maximum number of iterations
        self.expanded_nodes = []

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        start_time = time.time()
        plan = []

        # start state as the root
        start_id = self.tree.add_vertex(self.planning_env.start)
        goal_reached = False

        for _ in range(self.max_iter):
            # sample a random state
            if np.random.rand() < self.goal_prob:
                rand_state = self.planning_env.goal
            else:
                rx = np.random.uniform(self.planning_env.xlimit[0], self.planning_env.xlimit[1])
                ry = np.random.uniform(self.planning_env.ylimit[0], self.planning_env.ylimit[1])
                rand_state = np.array([rx, ry])

            # nearest vertex to the sample
            near_id, near_state = self.tree.get_nearest_state(rand_state)

            # extend from the nearest vertex
            new_state = self.extend(near_state, rand_state)

            # check collisions
            if (not self.planning_env.state_validity_checker(new_state)) or \
                    (not self.planning_env.edge_validity_checker(near_state, new_state)):
                continue  # if invalid, skip

            # add new state
            new_id = self.tree.add_vertex(new_state)
            edge_cost = self.planning_env.compute_distance(near_state, new_state)
            self.tree.add_edge(near_id, new_id, edge_cost)

            # keep track
            self.expanded_nodes.append(new_state)

            # check if we reached the goal
            dist_to_goal = np.linalg.norm(new_state - self.planning_env.goal)
            if dist_to_goal < self.goal_threshold:
                plan = self.reconstruct_path(new_id)  # reconstruct the path and break
                goal_reached = True
                break

        print('Total time: {:.2f} seconds'.format(time.time() - start_time))

        # If we have a valid plan, compute cost
        if goal_reached:
            plan_cost = self.compute_cost(plan)
            print('Goal reached!')
        else:
            # If not reached, best we can do is the nearest-to-goal state so far
            print('Max iterations reached without fully connecting to goal.')
            plan = self.reconstruct_nearest_path_to_goal()
            plan_cost = self.compute_cost(plan)

        # Print stats
        print('Total cost of path: {:.2f}'.format(plan_cost))

        return np.array(plan)

    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''

        if self.ext_mode == 'E1':
            # Extend fully to the random state
            return rand_state

        elif self.ext_mode == 'E2':
            # Move in direction from near_state to rand_state, with step size self.eta
            direction = rand_state - near_state
            dist = np.linalg.norm(direction)
            if dist < 1e-9:
                # The random sample is basically the same as near_state
                return near_state
            direction = direction / dist  # unit vector
            # if distance is less than eta, go directly to rand_state
            step = min(dist, self.eta)
            new_state = near_state + step * direction
            return new_state

        else:
            raise ValueError(f"Unknown ext_mode {self.ext_mode}")

    def reconstruct_path(self, goal_id):
        """
        Traverse back from the goal to the start using self.tree.edges.
        Returns a list of states from start -> goal.
        """
        path = []
        current_id = goal_id
        while True:
            current_vertex = self.tree.vertices[current_id]
            path.append(current_vertex.state)
            if current_id == 0:
                # Reached the root (start) which is ID = 0
                break
            current_id = self.tree.edges[current_id]

        path.reverse()
        return path

    def reconstruct_nearest_path_to_goal(self):
        """
        If we did not exactly reach the goal, try to find the tree vertex
        that is closest to the real goal and backtrack from there.
        """
        best_id = None
        min_dist = float('inf')
        goal = self.planning_env.goal

        # Find the vertex that is closest to the goal
        for vid, vertex in self.tree.vertices.items():
            d = np.linalg.norm(vertex.state - goal)
            if d < min_dist:
                min_dist = d
                best_id = vid

        # reconstruct path from that vertex
        return self.reconstruct_path(best_id)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        if len(plan) < 2:
            return 0.0
        cost = 0.0
        for i in range(len(plan) - 1):
            cost += np.linalg.norm(plan[i + 1] - plan[i])
        return cost

    def get_expanded_nodes(self):
        """
        Return the list of visited/expanded states for optional plotting.
        """
        return self.expanded_nodes