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
        self.num_of_iter = 1000
        self.term = "path_exist"

    def shortest_way(self):
        way = []
        end = self.planning_env.goal
        end_idx = self.tree.get_idx_for_state(end)
        begin = self.planning_env.start
        if self.tree.get_idx_for_state(end) is not None:
            curr_idx = end_idx
            way.append(self.tree.vertices[end_idx])
            while curr_idx is not None and not np.array_equal(way[-1].state, begin):
                way.append(self.tree.vertices[self.tree.edges[curr_idx]])
                curr_idx = self.tree.edges[curr_idx]
            return list(reversed(way))
        return None

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''
        ''
        begin = time.time()
        self.tree.add_vertex(self.planning_env.start)
        if not self.term == 'path_exist':
            for j in range(1, self.num_of_iter):
                random_x = self.get_rand()
                _, close_x = self.tree.get_nearest_state(random_x)
                new_x = self.extend(close_x, random_x)
                if self.tree.get_idx_for_state(random_x) is None and self.planning_env.edge_validity_checker(close_x,new_x):
                    self.tree.add_vertex(new_x)
                    self.tree.add_edge(self.tree.get_idx_for_state(close_x),self.tree.get_idx_for_state(new_x),self.planning_env.compute_distance(new_x,close_x))
        else:
            while self.shortest_way() is None:
                random_x = self.get_rand()
                _, close_x = self.tree.get_nearest_state(random_x)
                new_x = self.extend(close_x, random_x)
                if self.tree.get_idx_for_state(random_x) is None and self.planning_env.edge_validity_checker(close_x,
                                                                                                             new_x):
                    self.tree.add_vertex(new_x)
                    self.tree.add_edge(self.tree.get_idx_for_state(close_x), self.tree.get_idx_for_state(new_x),
                                       self.planning_env.compute_distance(new_x, close_x))

        print('cost: {:.2f}'.format(self.compute_cost(self.shortest_way())))
        print('time: {:.2f}'.format(time.time()-begin))
        final_plan = list(map(lambda vertex: vertex.state, self.shortest_way()))
        return time.time()-begin, self.compute_cost(self.shortest_way()), np.array(final_plan)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps.
        @param plan A given plan for the robot.
        '''
        if (plan is not None):
            return plan[-1].cost
        return 0


    def extend(self, near_state, rand_state):
        '''
        Compute and return a new position for the sampled one.
        @param near_state The nearest position to the sampled position.
        @param rand_state The sampled position.
        '''
        if (self.ext_mode == 'E2'):
            way_len = np.linalg.norm(rand_state - near_state)
            param_etha = 15
            if not param_etha > way_len:
                return near_state + param_etha * (rand_state - near_state) / np.linalg.norm(rand_state - near_state)
            return rand_state

        if(self.ext_mode == 'E1'):
            return rand_state


    def get_rand(self):
        if not self.goal_prob > random.uniform(0, 100) / 100:
            position = np.array(
                [random.uniform(0, self.planning_env.xlimit[1]), random.uniform(0, self.planning_env.ylimit[1])])
            while (self.planning_env.state_validity_checker(position) == False):
                position = np.array(
                    [random.uniform(0, self.planning_env.xlimit[1]), random.uniform(0, self.planning_env.ylimit[1])])
            return position
        else:
            return self.planning_env.goal