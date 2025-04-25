from collections import deque
import numpy as np


class DependencyGraph:
    def __init__(self, start_arr, goal_arr, object_size):
        '''
        Args:
        start_arr: start arrangement. key: int obj ID, value: (x,y) coordinate
        goal_arr: goal arrangement. key: int obj ID, value: (x,y) coordinate
        object_width: width of the object
        object_length: length of the object

        Returns:
        dependency graph DG: DG[objID]=set(obj1, obj2, ...)
        '''
        self.dependency_graph= {}
        for goal_obj_id, goal_center in goal_arr.items():
            self.dependency_graph[goal_obj_id] = set()
            for start_obj_id, start_center in start_arr.items():
                if start_obj_id == goal_obj_id:
                    continue
                # check if there's intersection with the square
                if( (abs(start_center[0] - goal_center[0]) < object_size) and
                    (abs(start_center[1] - goal_center[1]) < object_size)):
                    self.dependency_graph[goal_obj_id].add(start_obj_id)
        
        self.independent_sets = self.get_independent_sets()
        self.cycles = self.find_all_cycles()
        self.chains = self.find_all_chains()

    def get_dependency_graph(self):
        return self.dependency_graph
    
    def has_cycles(self):
        return len(self.cycles) > 0
    
    def has_chains(self):
        return len(self.chains) > 0
    
    def is_dependent(self, start, target):
        """
        Checks if the target node is dependent on the start node in the dependency graph.
        :param dependency_graph: Dictionary representing the adjacency list with integer node IDs.
        :param start: The starting node.
        :param target: The target node to check for dependency.
        :return: True if target is dependent on start, False otherwise.
        """
        visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()
            if node == target:
                return True
            if node not in visited:
                visited.add(node)
                queue.extend(self.dependency_graph.get(node, []))

        return False
    
    def is_directly_dependent(self, start, target):
        """
        Checks if the target node is directly dependent on the start node (direct neighbors).
        :param dependency_graph: Dictionary representing the adjacency list with integer node IDs.
        :param start: The starting node.
        :param target: The target node to check for direct dependency.
        :return: True if target is a direct neighbor of start, False otherwise.
        """
        return target in self.dependency_graph.get(start, [])

    def get_independent_sets(self):
        """
        Nodes with no incoming edges (no dependencies) OR no outgoing edges (no one depends on them).
        """
        independent_sets = []
        all_nodes = set(self.dependency_graph.keys())
        # Collect nodes that have incoming edges
        nodes_with_incoming = set()
        for deps in self.dependency_graph.values():
            nodes_with_incoming.update(deps)

        for node in all_nodes:
            has_incoming = node in nodes_with_incoming
            has_outgoing = len(self.dependency_graph[node]) > 0
            if not has_incoming or not has_outgoing:
                independent_sets.append(node)
        
        return independent_sets

    
    def get_cycles(self):
        return self.cycles
    
    def get_chains(self):
        return self.chains

    def find_all_cycles(self):
        """
        Returns all cycles in a directed graph.
        :param dependency_graph: Dictionary representing the adjacency list with integer node IDs.
        :return: List of cycles (each cycle as a list of node IDs).
        """
        cycles = []
        path = []
        visited = set()

        def dfs(node, start_node):
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])  # Extract the cycle
                return

            path.append(node)
            for neighbor in self.dependency_graph.get(node, []):
                if neighbor == start_node or neighbor not in visited:
                    dfs(neighbor, start_node)

            path.pop()
            visited.add(node)

        for node in self.dependency_graph:
            if node not in visited:
                dfs(node, node)

        return cycles
    
    def find_all_chains(self):
        """
        Returns all distinct chains (linear dependency paths) in the dependency graph
        that do NOT include any nodes that are part of a cycle.
        
        Chains that include cycle nodes are not returned (because cycles should be handled first).
        """
        chains = []
        visited = set()

        # Compute the set of all nodes that belong to any cycle.
        cycles = self.find_all_cycles()
        cycle_nodes = set()
        for cyc in cycles:
            # In case cycles are returned as lists
            if isinstance(cyc, list):
                cycle_nodes.update(cyc)
            else:
                cycle_nodes.add(cyc)

        def dfs(node, path):
            if node in path:  # A cycle is detected; abort this branch.
                return
            path.append(node)
            visited.add(node)
            # If node has no outgoing edges (i.e. terminal) and path length > 1, consider it a complete chain.
            # Only add the chain if none of its nodes are in a cycle.
            if not self.dependency_graph.get(node, []) and len(path) > 1:
                if not set(path).intersection(cycle_nodes):
                    chains.append(path[:])
            else:
                for neighbor in self.dependency_graph.get(node, []):
                    dfs(neighbor, path)
            path.pop()

        for node in self.dependency_graph:
            if node not in visited:
                dfs(node, [])

        return chains
