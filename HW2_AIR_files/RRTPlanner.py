import numpy as np
from RRTTree import RRTTree
import time
import random

class RRTPlanner(object):

    def __init__(self, planning_env, ext_mode, goal_prob):

        # set environment and search tree
        self.planning_env = planning_env
        self.tree = RRTTree(self.planning_env)

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

    def find_path(self):
        """
        path_start/goal = states that exist on the tree
        :return: returns the shortest path available if exists
        """

        path = []
        path_state = []

        start = self.planning_env.start
        goal = self.planning_env.goal

        goal_vid = self.tree.get_idx_for_state(goal)
        if self.tree.get_idx_for_state(goal) is None:
            # print("here :(")
            return None

        path.append(self.tree.vertices[goal_vid])
        current_vid = goal_vid
        while (not np.array_equal(path[-1].state, start)) and current_vid is not None:
            # print(f" path[-1].state = {path[-1].state} , start = {start}")
            vid_parent = self.tree.edges[current_vid]
            path.append(self.tree.vertices[vid_parent])
            # print(f"vid_parent {self.tree.vertices[vid_parent].state}")
            current_vid = vid_parent

        # path_state = [vertex.state for vertex in path]
        return path[::-1]

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        start_time = time.time()

        # initialize an empty plan.
        plan = []

        # initialize tree
        self.tree.add_vertex(self.planning_env.start)
        # first let's build the rrt tree
        if (self.algo_termination == 'path_exist'):
            while (self.find_path() is None):
                x_rand = self.sample_random_state()
                _,x_near = self.tree.get_nearest_state(x_rand)
                x_new = self.extend(x_near, x_rand)
                if (self.planning_env.edge_validity_checker(x_near,x_new) and self.tree.get_idx_for_state(x_rand) is None):
                    self.tree.add_vertex(x_new)
                    idx_x_near = self.tree.get_idx_for_state(x_near)
                    idx_x_new = self.tree.get_idx_for_state(x_new)
                    edge_cost = self.planning_env.compute_distance(x_new,x_near)
                    self.tree.add_edge(idx_x_near,idx_x_new,edge_cost)
        elif (self.algo_termination == 'epoch_time'):
            for i in range (1,self.n_iteration):
                x_rand = self.sample_random_state()
                _,x_near = self.tree.get_nearest_state(x_rand)
                x_new = self.extend(x_near, x_rand)
                if (self.planning_env.edge_validity_checker(x_near,x_new) and self.tree.get_idx_for_state(x_rand) is None):
                    self.tree.add_vertex(x_new)
                    idx_x_near = self.tree.get_idx_for_state(x_near)
                    idx_x_new = self.tree.get_idx_for_state(x_new)
                    edge_cost = self.planning_env.compute_distance(x_new,x_near)
                    self.tree.add_edge(idx_x_near,idx_x_new,edge_cost)
            print(f"path returned is : {self.find_path()}")


        plan = self.find_path()

        # print total path cost and time
        cost = self.compute_cost(plan)
        time_ = time.time()-start_time
        print('Total cost of path: {:.2f}'.format(cost))
        print('Total time: {:.2f}'.format(time_))
        plan = [vertex.state for vertex in plan]
        return time_,cost,np.array(plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        if (plan is None):
            return 0
        return plan[-1].cost

    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''

        if(self.ext_mode == 'E1'):
            return rand_state

        if (self.ext_mode == 'E2'):
            etha = ((self.planning_env.xlimit[1]+self.planning_env.ylimit[1])/2)/50

            # print(f"near_state {near_state}")
            # print(f"rand_state {rand_state}")

            path_length = np.linalg.norm(rand_state - near_state)
            if(etha > path_length):
                return rand_state

            # Calculate the direction vector from A to B
            direction = rand_state - near_state
            # Normalize the direction vector
            direction = direction / np.linalg.norm(direction)
            # Calculate the new point along the line
            new_point = near_state + etha * direction
            return new_point

    def sample_random_state(self):

        random_state_choose = random.uniform(0, 100)
        random_state_choose = random_state_choose / 100

        if (random_state_choose < self.bias):
            return self.planning_env.goal
        else:
            X_x = self.planning_env.xlimit[1]
            X_y = self.planning_env.ylimit[1]

            rand_state_x = random.uniform(0, X_x)
            rand_state_y = random.uniform(0, X_y)

            state = np.array([rand_state_x, rand_state_y])

            while (self.planning_env.state_validity_checker(state) == False):
                X_x = self.planning_env.xlimit[1]
                X_y = self.planning_env.ylimit[1]

                rand_state_x = random.uniform(0, X_x)
                rand_state_y = random.uniform(0, X_y)

                state = np.array([rand_state_x, rand_state_y])

            return state