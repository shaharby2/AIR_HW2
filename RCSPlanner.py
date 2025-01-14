import numpy as np
from typing import Optional
import heapq


class RCSPlanner(object):    
    def __init__(self, planning_env):
        self.planning_env = planning_env

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []

    def plan(self):
        '''
        Compute and return the plan as a numpy array of (x,y) states.
        '''

        # -- 1) Define coarse and fine action sets --
        coarse_actions = [(2, 0), (-2, 0), (0, 2), (0, -2),
                          (2, 2), (2, -2), (-2, 2), (-2, -2)]
        fine_actions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # -- 2) Initialize root node --
        start_state = self.planning_env.start
        root_node = Node(state=start_state, rank=0, resolution='coarse', parent=None)

        # Priority queue (min-heap) for the OPEN list, sorted by rank
        # Each entry will be (rank, unique_id, node).
        # The 'unique_id' can just be a counter to avoid ties in the heap.
        open_list = []
        counter = 0  # increments every time we push into the heap

        # Put root node in OPEN
        heapq.heappush(open_list, (root_node.rank, counter, root_node))
        counter += 1

        # CLOSED set to store states we have already visited
        closed_set = set()

        # -- 3) RCS main loop --
        while len(open_list) > 0:
            # Extract node with minimal rank
            _, _, current_node = heapq.heappop(open_list)

            # Check if current node's state is valid
            if not self.planning_env.state_validity_checker(current_node.state):
                continue

            # Check for duplicates
            state_tuple = tuple(current_node.state)
            if state_tuple in closed_set:
                continue

            # Mark visited
            closed_set.add(state_tuple)
            self.expanded_nodes.append(current_node.state)

            # Check if we have reached the goal
            if np.allclose(current_node.state, self.planning_env.goal):
                return self.reconstruct_path(current_node)

            # 3a) Coarse expansions from the current node
            for action in coarse_actions:
                new_state = current_node.state + np.array(action)

                # Check edge validity
                if not self.planning_env.edge_validity_checker(current_node.state, new_state):
                    continue

                new_node = Node(
                    state=new_state,
                    rank=current_node.rank + 1,
                    resolution='coarse',
                    parent=current_node
                )

                heapq.heappush(open_list, (new_node.rank, counter, new_node))
                counter += 1

            # 3b) Fine expansions from the *parent* if resolution = coarse (and not root)
            #     as per the pseudocode:
            #
            #     If v != root AND v.resolution = coarse:
            #         For each action in fineSet:
            #             newNode = propagate(v.parent, action)
            #
            if current_node.parent is not None and current_node.resolution == 'coarse':
                parent_node = current_node.parent
                for action in fine_actions:
                    new_state = parent_node.state + np.array(action)

                    if not self.planning_env.edge_validity_checker(parent_node.state, new_state):
                        continue

                    new_node = Node(
                        state=new_state,
                        rank=current_node.rank + 1,
                        resolution='fine',
                        parent=parent_node
                    )

                    heapq.heappush(open_list, (new_node.rank, counter, new_node))
                    counter += 1

        # If we exhaust OPEN without finding a goal, return empty
        return np.array([])

    def reconstruct_path(self, node):
        '''
        Reconstruct the path from the goal to the start using parent pointers.
        # YOU DON'T HAVE TO USE THIS FUNCTION!!!
        '''
        path = []
        big_steps = 0
        small_steps = 0
        total_length = 0.0
        while node:
            path.append(tuple(node.state))
            if node.resolution == 'coarse':
                big_steps += 1
            elif node.resolution == 'fine':
                small_steps += 1
            if node.parent:
                total_length += np.linalg.norm(node.state - node.parent.state)
            node = node.parent

        path.reverse()
        formatted_path = " -> ".join(map(str, path))
        print(f"Path: {formatted_path}")
        print(f"Total length: {total_length}")
        print(f"Total big steps (coarse): {big_steps}, {big_steps / (big_steps + small_steps) * 100:.2f}% of the path" )
        print(f"Total small steps (fine): {small_steps}, {small_steps / (big_steps + small_steps) * 100:.2f}% of the path" )
        return np.array(path)

    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        DO NOT MODIFY THIS FUNCTION!!!
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes

class Node:
    def __init__(self, state: tuple[int, int], rank: int, resolution: str, parent: Optional['Node'] = None):
        self.state: tuple[int, int] = state
        self.rank: int = rank
        self.resolution: str = resolution
        self.parent: Optional['Node'] = parent
