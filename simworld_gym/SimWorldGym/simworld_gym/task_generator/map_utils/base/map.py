import random
import heapq
import json
import math
from collections import defaultdict
from typing import Optional, List
from simworld_gym.task_generator.map_utils.utils.types import Vector, Road
from simworld_gym.task_generator.map_utils.config import Config
from collections import deque
import random
import math

class Node:
    def __init__(self, position: Vector, type: str = "normal"):
        self.position = position
        self.type = type   # "normal" or "intersection"
        self.roads_assignment = []

    def get_roads_assignment_id(self, json_path="/home/xuhong_he/CityLayout-ProceduralGeneration/output_test/progen_world.json"):
        with open(json_path, 'r') as f:
            data = json.load(f)

        result_ids = []

        for road in self.roads_assignment:
            center = road.center
            matched = False
            for obj in data["nodes"]:
                location = obj.get("properties", {}).get("location", {})
                loc_x = location.get("x")
                loc_y = location.get("y")

                if loc_x is None or loc_y is None:
                    continue

                dx = center.x - loc_x
                dy = center.y - loc_y
                distance = math.hypot(dx, dy)

                tolerance=1.0
                if distance <= tolerance:
                    result_ids.append(obj["id"])
                    matched = True
                    break  # move to next road after match

            if not matched:
                result_ids.append(None)  # or log warning if needed

        return result_ids


    def __str__(self):
        return f"Node(position={self.position}, type={self.type})"

    def __repr__(self):
        return f"Node(position={self.position}, type={self.type})"

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.position == other.position

    def __lt__(self, other):
        return self.position.x < other.position.x if self.position.x != other.position.x else self.position.y < other.position.y

    def __hash__(self):
        return hash(self.position)

class Edge:
    def __init__(self, node1: Node, node2: Node):
        self.node1 = node1
        self.node2 = node2
        self.weight = node1.position.distance(node2.position)
        self.road_assignment = None

    def __str__(self):
        return f"Edge(node1={self.node1}, node2={self.node2}, distance={self.weight})"

    def __repr__(self):
        return f"Edge(node1={self.node1}, node2={self.node2}, distance={self.weight})"

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return ((self.node1.position == other.node1.position and
                self.node2.position == other.node2.position) or
                (self.node1.position == other.node2.position and
                self.node2.position == other.node1.position))

    def __hash__(self):
        if self.node1.position.x < self.node2.position.x or \
            (self.node1.position.x == self.node2.position.x and
            self.node1.position.y <= self.node2.position.y):
            pos1, pos2 = self.node1.position, self.node2.position
        else:
            pos1, pos2 = self.node2.position, self.node1.position
        return hash((pos1, pos2))

class Map:
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.adjacency_list = defaultdict(list)

    def __str__(self):
        return f"Nodes: {self.nodes}\nEdges: {self.edges}\n"

    def __repr__(self):
        return self.__str__()

    def add_node(self, node: Node):
        self.nodes.add(node)

    def add_edge(self, edge: Edge):
        self.edges.add(edge)
        for existed_node in self.nodes:
            if existed_node == edge.node1:
                edge.node1 = existed_node
            if existed_node == edge.node2:
                edge.node2 = existed_node
        self.adjacency_list[edge.node1].append(edge.node2)
        self.adjacency_list[edge.node2].append(edge.node1)

    def get_adjacency_list(self):
        return self.adjacency_list

    def get_adjacent_points(self, node: Node):
        points = [n.position for n in self.adjacency_list[node]]
        return points

    def has_edge(self, edge: Edge):
        return edge in self.edges

    def get_points(self):
        return [node.position for node in self.nodes]

    def get_nodes(self):
        return self.nodes

    def get_edge(self, node1: Node, node2: Node) -> Optional[Edge]:
        edge = Edge(node1, node2)
        if edge in self.edges:
            return edge
        return None

    def map_node_to_road(self, node: Node, road: Road):
        """
        Map a Node to a Road. A node can belong to multiple roads.
        """
        for existed_node in self.nodes:
            if existed_node == node:
                target_node = existed_node

        # if target_node.position.x == 21700 and target_node.position.y == -18300:
        #     print(111)
        #     if not hasattr(target_node, "roads_assignment"):
        #         target_node.roads_assignment = []
        #     if road not in target_node.roads_assignment:
        #         target_node.roads_assignment.append(road)
        #     print(id(target_node))
        #     print(target_node)
        #     print(target_node.roads_assignment[0].start, target_node.roads_assignment[0].end)
        # else:
        if not hasattr(target_node, "roads_assignment"):
            target_node.roads_assignment = []
        if road not in target_node.roads_assignment:
            target_node.roads_assignment.append(road)


    def map_edge_to_road(self, edge: Edge, road):
        """
        Map an Edge to a Road. An edge belongs to only one road.
        """
        for existed_edge in self.edges:
            if existed_edge == edge:
                target_edge = existed_edge
        target_edge.road_assignment = road


    def get_random_node(self, exclude_pos: Optional[List[Node]] = None):
        # get a random node that is not an intersection
        nodes = [node for node in self.nodes if node.type == "normal"]
        # nodes = list(self.nodes)
        if exclude_pos:
            nodes = [node for node in nodes if node not in exclude_pos]
        return random.choice(nodes)

    def get_random_node_with_distance(self, base_pos: List[Node], exclude_pos: Optional[List[Node]] = None, min_distance: float = 0, max_distance: float = 100000):
        # get a random node that is not an intersection and is at least min_distance away from any nodes in exclude_pos
        nodes = [node for node in self.nodes if node.type != "intersection"]
        if exclude_pos:
            nodes = [node for node in nodes if node not in exclude_pos]
        # get a random node that is at least min_distance away from any nodes in exclude_pos
        while True:
            node = random.choice(nodes)
            base_node = random.choice(base_pos)
            if node.position.distance(base_node.position) >= min_distance and node.position.distance(base_node.position) <= max_distance:
                return node

    def get_random_node_with_edge_distance(self, base_pos: List[Node], exclude_pos: Optional[List[Node]] = None, min_distance: float = 0, max_distance: float = 200):
        # get a random node that is at least min_distance away from any nodes in exclude_pos
        nodes = list(self.nodes)
        if exclude_pos:
            nodes = [node for node in nodes if node not in exclude_pos]
        base_node = random.choice(base_pos)
        if min_distance == max_distance:
            target_distance = min_distance
        else:
            target_distance = random.randint(min_distance, max_distance)

        queue = deque([(base_node, 0)])  # (node, distance) pairs
        visited = {base_node}
        best_distance_diff = float('inf')
        result_nodes = []

        while queue:
            current_node, current_distance = queue.popleft()
            
            distance_diff = abs(current_distance - target_distance)
            
            if distance_diff < best_distance_diff:
                best_distance_diff = distance_diff
                result_nodes = [current_node]
            elif distance_diff == best_distance_diff:
                result_nodes.append(current_node)

            for neighbor in self.adjacency_list[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_distance + 1))

        return result_nodes

    def get_supply_points(self):
        return [node.position for node in self.nodes if node.type == "supply"]

    def connect_adjacent_roads(self):
        """
        Connect nodes from adjacent roads that are close to each other
        """
        nodes = list(self.nodes)
        connection_threshold = Config.SIDEWALK_OFFSET * 2 + 100   # Reasonable threshold for connecting nearby nodes


        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                node1 = nodes[i]
                node2 = nodes[j]

                # If nodes are close enough and not already connected
                if (node1.position.distance(node2.position) < connection_threshold and
                    not self.has_edge(Edge(node1, node2))):
                    self.add_edge(Edge(node1, node2))

    def interpolate_nodes(self, num_points: int = -1):
        """
        Interpolate nodes between existing nodes to create a smoother map
        """
        current_edges = list(self.edges)
        method = True if num_points == -1 else False


        for edge in current_edges:
            distance = edge.weight
            road = edge.road_assignment
            if method:
                num_points = int(distance / (2 * Config.SIDEWALK_OFFSET))

            if num_points <= 1:
                continue

            direction = (edge.node2.position - edge.node1.position).normalize()

            new_nodes = []
            
            supply_point_index = random.randint(2, num_points - 2) if num_points > 1 else None

            for i in range(1, num_points + 1):
                if method:
                    new_point = edge.node1.position + direction * (i * 2 * Config.SIDEWALK_OFFSET)
                else:
                    new_point = edge.node1.position + direction * (i * distance / num_points)
                node_type = "normal"
                new_node = Node(new_point, type=node_type)
                self.add_node(new_node)
                self.map_node_to_road(new_node, road)
                new_nodes.append(new_node)

            self.edges.remove(edge)
            self.adjacency_list[edge.node1].remove(edge.node2)
            self.adjacency_list[edge.node2].remove(edge.node1)

            self.map_node_to_road(edge.node1, road)
            self.map_node_to_road(edge.node1, road)

            all_nodes = [edge.node1] + new_nodes + [edge.node2]
            for i in range(len(all_nodes) - 1):
                self.add_edge(Edge(all_nodes[i], all_nodes[i + 1]))

    def get_edge_distance_between_two_points(self, point1: Node, point2: Node) -> int:
        """Calculate the minimum edge distance between two points using BFS.
        Args:
            point1: Starting node
            point2: Target node

        Returns:
            The minimum number of edges between the two points
        """
        if point1 == point2:
            return 0

        queue = deque([(point1, 0)])  # (node, distance) pairs
        visited = {point1}

        while queue:
            current_point, distance = queue.popleft()

            # Check if we've reached the target
            if current_point == point2:
                return distance

            # Explore neighbors
            for neighbor in self.adjacency_list[current_point]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        # If we get here, no path was found
        raise ValueError(f"No path found between {point1} and {point2}")


    def shortest_path(self, start_node, end_node):
        """
        Use Dijkstra algorithm to obtain the shortest path from start_node to end_node

        Args:
            start_node (Node)
            end_node (Node)

        Returns:
            path: nodes list of shortest path [node_1, node_2, ..., node_n]
            total_distance: the total distance of this path
        """
        for node in self.nodes:
            if node == start_node:
                start_node = node

        pq = []
        heapq.heappush(pq, (0, start_node, []))

        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0

        predecessors = {}

        while pq:
            current_distance, current_node, path = heapq.heappop(pq)

            if current_node in path:
                continue
            path = path + [current_node]
            if current_node == end_node:
                return path, current_distance

            for neighbor in self.adjacency_list.get(current_node, []):
                edge = self.get_edge(current_node, neighbor)
                if edge is None:
                    continue
                # if neighbor.position.x == 21700 and neighbor.position.y == -18300:
                #     print(222)
                #     print(current_node)
                #     print(id(neighbor))

                new_distance = current_distance + edge.weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (new_distance, neighbor, path))
        return None, float('inf') 

    def find_path_with_less_turns(self, budget_turns, target_total_length):
        """
        Try to find a path that:
        - Has exactly target_total_length steps (edges)
        - Has at most budget_turns direction changes
        - Prefer moving straight without turning when possible

        Returns:
            path: list of nodes if such path exists, otherwise None
            total_edge_distance: sum of Euclidean distances between consecutive nodes
        """
        print("begin to find path with less turns")
        shuffled_nodes = list(self.nodes)
        random.shuffle(shuffled_nodes)
        for start_node in shuffled_nodes:
            if start_node.type == "intersection":
                continue
            visited = set()
            visited.add(start_node)
            path = [start_node]
            result = self._dfs_greedy_find(start_node, None, 0, 0, visited, path, budget_turns, target_total_length)
            if result is not None:
                if result[-1].type == "intersection":
                    continue
                final_path = result
                print(final_path)
                total_distance = self._calculate_total_distance(final_path)
                return final_path, total_distance
        return None, 0.0

    def _dfs_greedy_find(self, current_node, prev_direction, length_so_far, turns_so_far, visited, path, budget_turns, target_total_length):
        if length_so_far > target_total_length or turns_so_far > budget_turns:
            return None
        if length_so_far == target_total_length:
            return list(path)

        neighbors = self.adjacency_list.get(current_node, [])

        straight_neighbors = []
        turn_neighbors = []

        for neighbor in neighbors:
            if neighbor in visited:
                continue

            move_direction = (
                neighbor.position.x - current_node.position.x,
                neighbor.position.y - current_node.position.y
            )
            move_direction = self._normalize_direction(move_direction)

            if prev_direction is None or move_direction == prev_direction:
                straight_neighbors.append((neighbor, move_direction))
            else:
                turn_neighbors.append((neighbor, move_direction))

        # 如果budget_turns == 0，只能直走
        if budget_turns - turns_so_far <= 0:
            # 只能直走
            neighbor_list = straight_neighbors
        else:
            # 有一定概率直走
            prob_go_straight = 0.9  # 自定义概率，比如90%直走，10%拐弯
            if random.random() < prob_go_straight:
                neighbor_list = straight_neighbors
            else:
                neighbor_list = turn_neighbors

        # 在neighbor_list里尝试
        for neighbor, move_direction in neighbor_list:
            visited.add(neighbor)
            path.append(neighbor)
            new_turns = turns_so_far
            if prev_direction is not None and move_direction != prev_direction:
                new_turns += 1
            result = self._dfs_greedy_find(neighbor, move_direction, length_so_far + 1, new_turns, visited, path, budget_turns, target_total_length)
            if result is not None:
                return result
            path.pop()
            visited.remove(neighbor)

        # 如果上面没找到（比如直走没有可以走的），可以考虑另一组（比如先尝试直走失败，再尝试拐弯）
        if budget_turns - turns_so_far > 0 and neighbor_list != turn_neighbors and turn_neighbors:
            for neighbor, move_direction in turn_neighbors:
                visited.add(neighbor)
                path.append(neighbor)
                new_turns = turns_so_far + 1
                result = self._dfs_greedy_find(neighbor, move_direction, length_so_far + 1, new_turns, visited, path, budget_turns, target_total_length)
                if result is not None:
                    return result
                path.pop()
                visited.remove(neighbor)

        return None

    def _normalize_direction(self, direction):
        dx, dy = direction
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        return (dx, dy)

    def _calculate_total_distance(self, path):
        """
        Calculate the total Euclidean distance along a given path.
        """
        total = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1].position.x - path[i].position.x
            dy = path[i+1].position.y - path[i].position.y
            distance = math.sqrt(dx * dx + dy * dy)
            total += distance
        return total

    def _dfs_greedy_find(self, current_node, prev_direction, length_so_far, turns_so_far, visited, path, budget_turns, target_total_length):
        if length_so_far > target_total_length or turns_so_far > budget_turns:
            return None
        if length_so_far == target_total_length:
            return list(path)

        neighbors = self.adjacency_list.get(current_node, [])

        straight_neighbors = []
        turn_neighbors = []

        for neighbor in neighbors:
            if neighbor in visited:
                continue

            move_direction = (
                neighbor.position.x - current_node.position.x,
                neighbor.position.y - current_node.position.y
            )
            move_direction = self._normalize_direction(move_direction)

            if prev_direction is None or move_direction == prev_direction:
                straight_neighbors.append((neighbor, move_direction))
            else:
                turn_neighbors.append((neighbor, move_direction))

        # 如果budget_turns == 0，只能直走
        if budget_turns - turns_so_far <= 0:
            # 只能直走
            neighbor_list = straight_neighbors
        else:
            # 有一定概率直走
            prob_go_straight = 0.9  # 自定义概率，比如90%直走，10%拐弯
            if random.random() < prob_go_straight:
                neighbor_list = straight_neighbors
            else:
                neighbor_list = turn_neighbors

        # 在neighbor_list里尝试
        for neighbor, move_direction in neighbor_list:
            visited.add(neighbor)
            path.append(neighbor)
            new_turns = turns_so_far
            if prev_direction is not None and move_direction != prev_direction:
                new_turns += 1
            result = self._dfs_greedy_find(neighbor, move_direction, length_so_far + 1, new_turns, visited, path, budget_turns, target_total_length)
            if result is not None:
                return result
            path.pop()
            visited.remove(neighbor)

        # 如果上面没找到（比如直走没有可以走的），可以考虑另一组（比如先尝试直走失败，再尝试拐弯）
        if budget_turns - turns_so_far > 0 and neighbor_list != turn_neighbors and turn_neighbors:
            for neighbor, move_direction in turn_neighbors:
                visited.add(neighbor)
                path.append(neighbor)
                new_turns = turns_so_far + 1
                result = self._dfs_greedy_find(neighbor, move_direction, length_so_far + 1, new_turns, visited, path, budget_turns, target_total_length)
                if result is not None:
                    return result
                path.pop()
                visited.remove(neighbor)

        return None
