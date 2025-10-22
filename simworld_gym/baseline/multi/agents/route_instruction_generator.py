from typing import List, Dict, Any
import numpy as np
import os
import json
import random
from typing import Optional

from simworld.citygen.route.route_generator import RouteGenerator
from simworld.citygen.function_call.city_function_call import CityFunctionCall
from simworld.citygen.route.route_manager import RouteManager
from simworld.citygen.dataclass.dataclass import Point
from simworld.utils.data_importer import DataImporter

import json
from simworld_gym import Config, Vector, Road, Node, Edge, Map
from simworld_gym.task_generator.map_utils.utils.utils import visualize_map_with_path

from gpt4o_wrapper import GPT4oVision


@staticmethod
def calculate_correct_robot_orientation_action(robot_current_orientation, correct_robot_orientation, tolerance=5):
    # 归一化角度到 [0, 360)
    robot_current_orientation %= 360
    correct_robot_orientation %= 360

    # 计算角度差（顺时针方向）
    delta = (correct_robot_orientation - robot_current_orientation) % 360

    if abs(delta) <= tolerance or abs(delta - 360) <= tolerance:
        return []
    elif abs(delta - 90) <= tolerance:
        return [4]
    elif abs(delta - 180) <= tolerance:
        return [4, 4]
    elif abs(delta - 270) <= tolerance:
        return [5]
    else:
        raise ValueError(f"Angle difference {delta} not within expected ranges.")

@staticmethod
def calculate_robot_location_after_move_forwards(current_robot_location, orientation, num_move_forwards):
    if orientation == 0:
        return [current_robot_location[0] + num_move_forwards * 500, current_robot_location[1]]
    elif orientation == 90:
        return [current_robot_location[0], current_robot_location[1] + num_move_forwards * 500]
    elif orientation == 180:
        return [current_robot_location[0] - num_move_forwards * 500, current_robot_location[1]]
    elif orientation == 270:
        return [current_robot_location[0], current_robot_location[1] - num_move_forwards * 500]
    else:
        raise ValueError(f"Invalid orientation: {orientation}")

@staticmethod
def calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, correct_robot_location):
    if current_robot_orientation == 0:
        num_move_forwards = round((correct_robot_location[0]-current_robot_location[0])/500)
    elif current_robot_orientation == 90:
        num_move_forwards = round((correct_robot_location[1]-current_robot_location[1])/500)
    elif current_robot_orientation == 180:
        num_move_forwards = round((correct_robot_location[0]-current_robot_location[0])/500)
    elif current_robot_orientation == 270:
        num_move_forwards = round((correct_robot_location[1]-current_robot_location[1])/500)
    else:
        raise ValueError(f"Invalid orientation: {current_robot_orientation}")
    current_robot_location = calculate_robot_location_after_move_forwards(current_robot_location, current_robot_orientation, num_move_forwards)
    return [0] * num_move_forwards, current_robot_location

@staticmethod
def calculate_correct_robot_turning_action(current_robot_location, current_robot_orientation, correct_robot_location, correct_robot_orientation, tolerance=5):
    action_before_turn, current_robot_location = calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, correct_robot_location)
    turn_action = calculate_correct_robot_orientation_action(current_robot_orientation, correct_robot_orientation, tolerance)
    current_robot_orientation = correct_robot_orientation
    action_after_turn, current_robot_location = calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, correct_robot_location)
    output_action = action_before_turn + turn_action + action_after_turn
    return output_action, current_robot_location, current_robot_orientation

@staticmethod
def calculate_correct_robot_finish_action(current_robot_location, current_robot_orientation, correct_robot_location, correct_robot_orientation, tolerance=5):
    action_before_turn, current_robot_location = calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, correct_robot_location)
    output_action = action_before_turn
    return output_action, current_robot_location, current_robot_orientation

class RouteInstructionGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the route generator."""
        self.seed = seed
        self.static_description = self._load_static_description()
        self.updated_map = None
        random.seed(seed)

    def _load_static_description(self):
        """Load static description from the template JSON file."""
        json_path = os.path.join(os.path.dirname(__file__), 'data', 'BuildingsDescriptionMapTemplate.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            template_library = json.load(f)
        return template_library

    def _load_map(self):
        """Load map from the template JSON file."""
        self.progen_city = json.load(open(self.city_path))["nodes"]
        self.city = DataImporter.import_city_data(self.city_path)
        self.cfg = CityFunctionCall(self.city_path, self.city)
        self.city_route_generator = RouteGenerator(RouteManager())
        self.basic_map = self._load_strctured_map()

    def _load_strctured_map(self):
        '''
        Initialize the platform, customers, delivery men, and stores from the map
        '''
        ############# load roads #############
        with open(self.road_path, 'r') as f:
            roads_data = json.load(f)

        roads = roads_data['roads']
        map = Map()
        road_objects = []
        for road in roads:
            start = Vector(road['start']['x']*100, road['start']['y']*100)
            end = Vector(road['end']['x']*100, road['end']['y']*100)
            road_objects.append(Road(start, end))

        # Initialize the map
        for road in road_objects:
            normal_vector = Vector(road.direction.y, -road.direction.x)
            point1 = road.start - normal_vector * (Config.SIDEWALK_OFFSET) + road.direction * Config.SIDEWALK_OFFSET
            point2 = road.end - normal_vector * (Config.SIDEWALK_OFFSET) - road.direction * Config.SIDEWALK_OFFSET

            point3 = road.end + normal_vector * (Config.SIDEWALK_OFFSET) - road.direction * Config.SIDEWALK_OFFSET
            point4 = road.start + normal_vector * (Config.SIDEWALK_OFFSET) + road.direction * Config.SIDEWALK_OFFSET

            node1 = Node(point1, "intersection")
            node2 = Node(point2, "intersection")
            node3 = Node(point3, "intersection")
            node4 = Node(point4, "intersection")

            map.add_node(node1)
            map.add_node(node2)
            map.add_node(node3)
            map.add_node(node4)
            map.map_node_to_road(node1, road)
            map.map_node_to_road(node2, road)
            map.map_node_to_road(node3, road)
            map.map_node_to_road(node4, road)

            map.add_edge(Edge(node1, node2))
            map.add_edge(Edge(node3, node4))
            map.add_edge(Edge(node1, node4))
            map.add_edge(Edge(node2, node3))
            map.map_edge_to_road(Edge(node1, node2), road)
            map.map_edge_to_road(Edge(node3, node4), road)

        # Connect adjacent roads
        map.connect_adjacent_roads()
        map.interpolate_nodes(num_points_on_road=6)
        return map
    
    def visualize_task(self, nodes: List[Dict[str, Any]], save_path: Optional[str] = None):
        """
        Visualize the task on the map.
        """
        nodes_of_interest = []
        path = [Node(Vector(node['pos'][0]*100, node['pos'][1]*100), node['type']) for node in nodes]
        
        if save_path:
            visualize_map_with_path(self.basic_map, path, nodes_of_interest, save_path=save_path)
        else:
            visualize_map_with_path(self.basic_map, path, nodes_of_interest)
    
    def _calculate_cross_product(self, current_node: Dict[str, Any], pre_node: Dict[str, Any], next_node: Dict[str, Any]) -> float:
        """
        Calculate the cross product of the current node and the next node.
        """
        # Vector representing the direction of arrival at 'node'
        current_direction = current_node['pos'] - pre_node['pos']
        # Vector from 'node' to 'next_node'
        next_direction = next_node['pos'] - current_node['pos']

        # Calculate the 2D cross product (z-component)
        cross_product_z = current_direction[0] * next_direction[1] - current_direction[1] * next_direction[0]

        # Define a threshold for floating point comparisons
        threshold = 1e-5 # Adjust sensitivity as needed

        # Determine direction
        if abs(cross_product_z) < threshold:
            # Treat all near-collinear cases as straight
            direction = "Straight"
        elif cross_product_z > 0:
            direction = "Right"
        else: # cross_product_z < 0
            direction = "Left"

        print("cross_product_z:", cross_product_z)
        print(f"Relative direction to next_node: {direction}")
        return direction
    
    def load_env(self, city_path: str, road_path: str):
        self.city_path = city_path
        self.road_path = road_path
        self._load_map()
    
    def generate_instructions(self, start: Node, end_node: Node, output_path: str="", if_count_intersections: bool = True, if_generate_instructions: bool = True) -> str:
        """
        Generate a task from the start and end points.
        """

        print(f"Finding shortest path between {start} and {end_node}")
        path, distance = self.updated_map.shortest_path(start, end_node)
        print(f"Found path with {len(path)} nodes")
        # Create subfolder for each end node
        nodes = []
        for i, p in enumerate(path):
            node = {
                'pos': np.array([p.position.x/100, p.position.y/100]) ,
                'type': p.type,
            }
            road_ids = p.get_roads_assignment_id(self.city_path)
            if len(road_ids) > 1:
                # Get next node's road IDs if this isn't the last node in path
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    next_road_ids = next_node.get_roads_assignment_id(self.city_path)
                    # Find intersection of current and next node's road IDs
                    common_ids = list(set(road_ids) & set(next_road_ids))
                    if common_ids:
                        road_ids = common_ids
            node['road'] = road_ids[0] if road_ids else None
            nodes.append(node)

        templates, nodes_of_interest, evaluation_of_subtask = self.nodes_to_templates(nodes, if_count_intersections)
        if if_generate_instructions:
            description = self.generate_description_clear_finishing_condition(templates, if_count_intersections)
        else:
            description = ""

        if output_path != "":
            os.makedirs(output_path, exist_ok=True)
            self.visualize_task(nodes, save_path=os.path.join(output_path, 'task.png'))

        return description, evaluation_of_subtask, nodes
    
    def generate_oracle_path(self, start: Node, end_node: Node, start_orientation: float = 0) -> List[Node]:
        """
        Generate the oracle path from the start to the end node.
        """
        print(f"Finding shortest path between {start} and {end_node}")
        path, distance = self.updated_map.shortest_path(start, end_node)
        print(f"Found path with {len(path)} nodes")
        # Create subfolder for each end node
        nodes = []
        for i, p in enumerate(path):
            node = {
                'pos': np.array([p.position.x/100, p.position.y/100]) ,
                'type': p.type,
            }
            road_ids = p.get_roads_assignment_id(self.city_path)
            if len(road_ids) > 1:
                # Get next node's road IDs if this isn't the last node in path
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    next_road_ids = next_node.get_roads_assignment_id(self.city_path)
                    # Find intersection of current and next node's road IDs
                    common_ids = list(set(road_ids) & set(next_road_ids))
                    if common_ids:
                        road_ids = common_ids
            node['road'] = road_ids[0] if road_ids else None
            nodes.append(node)

        _, _, evaluation_of_subtask = self.nodes_to_templates(nodes, False)
        current_robot_orientation = start_orientation % 360
        current_robot_location = [start.position.x, start.position.y]
        actions = []
        for i, evaluation in enumerate(evaluation_of_subtask):
            if evaluation['type'] == "ORIENTATION":
                subtask_actions = calculate_correct_robot_orientation_action(current_robot_orientation, evaluation['correct_robot_orientation'])
                current_robot_orientation = evaluation_of_subtask[i]['correct_robot_orientation'] % 360
            elif evaluation['type'] == "ROAD":
                subtask_actions, current_robot_location = calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, evaluation_of_subtask[1]['correct_robot_location'])
            elif evaluation['type'] == "TURNING":
                subtask_actions, current_robot_location, current_robot_orientation = calculate_correct_robot_turning_action(current_robot_location=current_robot_location,
                                                                                                                            current_robot_orientation=current_robot_orientation,
                                                                                                                            correct_robot_location=evaluation["correct_robot_location"],
                                                                                                                            correct_robot_orientation=evaluation["correct_robot_orientation"])
            elif evaluation['type'] == "FINISH":
                subtask_actions, current_robot_location, current_robot_orientation = calculate_correct_robot_finish_action(current_robot_location=current_robot_location,
                                                                                                                            current_robot_orientation=current_robot_orientation,
                                                                                                                            correct_robot_location=evaluation["correct_robot_location"],
                                                                                                                            correct_robot_orientation=evaluation["correct_robot_orientation"])                                             
            actions += subtask_actions
        return actions


    def nodes_to_templates(self, nodes: List[Dict[str, Any]], if_count_intersections: bool = True) -> List[str]:
        """
        Convert a sequence of nodes into templates.
        
        Args:
            nodes: List of dictionaries containing node information
                 Each node should have: pos, type, and road
            
        Returns:
            List of template strings
        """
        
        nodes_of_interest = []
        templates = []
        evaluation_of_subtask = []
        node_index = 0
        past_num_intersections = 0
        while node_index < len(nodes):
            current_node = nodes[node_index]
            if node_index == 0:
                # Generate Orientation template
                node_of_interest, cue_template = self._generate_cue_template(current_node, nodes[node_index + 1])
                # Extend the nodes_of_interest list with any new nodes
                nodes_of_interest.extend(node_of_interest)
                current_robot_orientation = node_of_interest[0]['robot_orientation'][0]
                template = {'cue': cue_template}
                template['direction'] = node_of_interest[0]['robot_orientation'][1]
                evaluation_of_subtask.append({"type": "ORIENTATION", "correct_robot_orientation": current_robot_orientation})
                template['template_type'] = "ORIENTATION"
                templates.append(template)
                node_index += 1
            elif current_node['type'] == 'intersection':
                direction, num_nodes_between = self._check_intersection_turning(node_index, nodes)
                if direction == "Straight":
                    past_num_intersections += 1
                elif direction == "Left" or direction == "Right":
                    # Generate intersection template, inform the agent the correct intersection to turn
                    template = self._generate_intersection_template(past_num_intersections, if_count_intersections)
                    node_of_interest, cue_template = self._generate_cue_template(nodes[node_index - 1], current_node)
                    node_of_interest[0]['robot_location'] = current_node['pos'] - (current_node['pos'] - nodes[node_index - 1]['pos']) / np.linalg.norm(current_node['pos'] - nodes[node_index - 1]['pos']) * 20
                    template['cue'] = cue_template
                    nodes_of_interest.extend(node_of_interest)
                    # Calculate the correct robot orientation and location
                    evaluation_of_subtask.append({"type": "ROAD", "correct_robot_location": [node_of_interest[0]['robot_location'][0]*100, node_of_interest[0]['robot_location'][1]*100]})
                    templates.append(template)

                    # Generate turning template, inform the agent the correct direction to turn
                    template = self._generate_turning_template(direction)
                    end_node = ((nodes[node_index + num_nodes_between]['pos'] - nodes[node_index + num_nodes_between - 1]['pos']) / np.linalg.norm(nodes[node_index + num_nodes_between]['pos'] - nodes[node_index + num_nodes_between - 1]['pos'])) * 20 + nodes[node_index + num_nodes_between]['pos']
                    begin_node = nodes[node_index]['pos']
                    # Create rectangle corners with some padding
                    bottom_left = (np.minimum(begin_node, end_node) - 10) * 100  # Expand 10 units in each direction
                    top_right = (np.maximum(begin_node, end_node) + 10) * 100
                    correct_robot_position_area = [bottom_left.tolist(), top_right.tolist()]
                    current_robot_orientation = self.calculate_orientation(nodes[node_index+num_nodes_between-1], nodes[node_index+num_nodes_between])[0]
                    evaluation_of_subtask.append({"type": "TURNING", 
                                                  "correct_robot_orientation": current_robot_orientation, 
                                                  "correct_robot_position_area": correct_robot_position_area, 
                                                  "correct_robot_location": [nodes[node_index+num_nodes_between-1]['pos'][0]*100, nodes[node_index+num_nodes_between-1]['pos'][1]*100]})
                    templates.append(template)
                    node_of_interest = [{
                        'robot_orientation': self.calculate_orientation(nodes[node_index+num_nodes_between-1], nodes[node_index+num_nodes_between]),
                        'robot_location': nodes[node_index+num_nodes_between-1]['pos']
                    }]
                    nodes_of_interest.extend(node_of_interest)
                    
                    past_num_intersections = 0
                else:
                    raise ValueError(f"Invalid direction: {direction}")
                node_index += num_nodes_between
            else:
                node_index += 1

        node_of_interest, template = self._generate_cue_template(nodes[-2], nodes[-1])
        current_robot_orientation = node_of_interest[0]['camera_orientation'][2]
        evaluation_of_subtask.append({"type": "FINISH", 
                                        "correct_robot_orientation": current_robot_orientation,
                                        "correct_robot_location": [node_of_interest[0]['front_door_position'].x * 100, node_of_interest[0]['front_door_position'].y * 100]})
        node_of_interest[0]['robot_location'] = [node_of_interest[0]['front_door_position'].x, node_of_interest[0]['front_door_position'].y]
        node_of_interest[0]['robot_orientation'] = [current_robot_orientation, ""]
        nodes_of_interest.extend(node_of_interest)
        template['template_type'] = "FINISH"
        templates.append(template)
        return templates, nodes_of_interest, evaluation_of_subtask

    def _check_intersection_turning(self, current_node_index: int, nodes: List[Dict[str, Any]]) -> bool:
        """
        Check the turning direction of the intersection.
        Returns:
            tuple: (direction, num_nodes_between)
            - direction: 1 for right turn, -1 for left turn, 0 for no turn (straight)
            - num_nodes_between: number of nodes between the current intersection and next road node
        """
        pre_road_node = nodes[current_node_index - 1]
        # find the next road node
        next_road_index = None
        for i in range(current_node_index + 1, len(nodes)):
            if nodes[i]['type'] == 'normal':
                next_road_index = i
                break
        next_road_node = nodes[next_road_index]
        num_nodes_between = next_road_index - current_node_index
        
        # Calculate direction before intersection
        pre_direction = np.array([
            nodes[current_node_index]['pos'][0] - pre_road_node['pos'][0],
            nodes[current_node_index]['pos'][1] - pre_road_node['pos'][1]
        ])
        
        # Calculate direction after intersection
        post_direction = np.array([
            next_road_node['pos'][0] - nodes[next_road_index-1]['pos'][0],
            next_road_node['pos'][1] - nodes[next_road_index-1]['pos'][1]
        ])
        
        # Normalize directions
        pre_direction = pre_direction / np.linalg.norm(pre_direction)
        post_direction = post_direction / np.linalg.norm(post_direction)
        
        
        # If directions are similar or the same, it's not a turn
        if abs(pre_direction[0] - post_direction[0]) < 1e-5 and abs(pre_direction[1] - post_direction[1]) < 1e-5:
            return "Straight", num_nodes_between
            
        # Otherwise calculate the turn direction
        direction = self._calculate_cross_product(pre_node=pre_road_node, current_node=nodes[next_road_index-1], next_node=next_road_node)
        
        return direction, num_nodes_between

    def _generate_cue_template(self, node, next_node, use_next_node_as_reference: bool = False) -> str:
        """
        Generate a template for the cue of the task.
        Note: Make sure the cue template is only used at normal nodes.
        """
        
        orientation = (next_node['pos'] - node['pos']) / np.linalg.norm(next_node['pos'] - node['pos'])
        # print("node:", node['pos'])
        # print("orientation:", orientation)
        
        building = None
        reference_point = Point(next_node['pos'][0], next_node['pos'][1]) if use_next_node_as_reference else Point(node['pos'][0], node['pos'][1])
        distance = 15
        direction_size = 2

        while building is None:
            # Get the reference point based on whether we're at the last node or not
           
            (_, building) = self.city_route_generator._get_point_along_direction(
                reference_point,
                self.city.city_quadtrees,
                orientation,
                direction_size,
                distance
            )
            direction_size -= 1
        
        # if building[0] not in landmark_building:
        #     return [], {}
        
        node_of_interest = {}
        building_description = {
            'type': self.static_description[building[0]].get('type', ''),
            'color': self.static_description[building[0]].get('color', ''),
            'height': self.static_description[building[0]].get('height', ''),
            'material': self.static_description[building[0]].get('material', ''),
            'extra_features': self.static_description[building[0]].get('extra_features', '')
        }

        # front_door_position = self._calculate_front_door_position(building[0], building[1])

        if abs(orientation[0]) <= 1e-5:
            if reference_point.x - building[1].x > 0:
                camera_orientation = [0, 0, 180]
                camera_position = Point((reference_point.x + 25), building[1].y )
                node_of_interest['position'] = Point((reference_point.x - 18), building[1].y)
                node_of_interest['front_door_position'] = Point((reference_point.x), building[1].y)
            else:
                camera_orientation = [0, 0, 0]
                camera_position = Point((reference_point.x - 25), building[1].y)
                node_of_interest['position'] = Point((reference_point.x + 18), building[1].y)
                node_of_interest['front_door_position'] = Point((reference_point.x), building[1].y)
        elif abs(orientation[1]) <= 1e-5:
            if reference_point.y - building[1].y > 0:
                camera_orientation = [0, 0, -90]
                camera_position = Point(building[1].x, (reference_point.y + 25))
                node_of_interest['position'] = Point(building[1].x, (reference_point.y - 18))
                node_of_interest['front_door_position'] = Point(building[1].x, (reference_point.y))
            else:
                camera_orientation = [0, 0, 90]
                camera_position = Point(building[1].x, (reference_point.y - 25))
                node_of_interest['position'] = Point(building[1].x, (reference_point.y + 18))
                node_of_interest['front_door_position'] = Point(building[1].x, (reference_point.y))
        node_of_interest['camera_orientation'] = camera_orientation
        node_of_interest['camera_position'] = camera_position

        node_of_interest['robot_orientation'] = self.calculate_orientation(node, next_node)
        node_of_interest['robot_location'] = [node_of_interest['front_door_position'].x - orientation[0] * 10, node_of_interest['front_door_position'].y - orientation[1] * 10]

        return [node_of_interest], {
            'object_description': building_description,
            'side': building[3]
        }

    def _generate_intersection_template(self, num_past_intersections: int, if_count_intersections: bool = True) -> str:
        """
        Generate a template string for an intersection.
        """
        def get_ordinal(n):
            if n <= 0:
                return ""
            if 10 <= n % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
            return str(n) + suffix

        current_intersection = num_past_intersections + 1
        if if_count_intersections:
            return {
                'template_type': "ROAD",
                'intersection': get_ordinal(current_intersection)
            }
        else:
            return {
                'template_type': "ROAD"
            }
    
    def _generate_turning_template(self, direction: str) -> str:
        """
        Generate a template string for a turning.
        """
        return {
            'template_type': "TURNING",
            'direction': direction
        }
    
    
    
    def generate_description_clear_finishing_condition(self, templates: List[str], using_number_of_intersections: bool = True) -> str:
        """
        Generate a natural language description from templates.
        
        Args:
            templates: List of template strings
            
        Returns:
            Natural language description
        """
        model = GPT4oVision(backend="openai", model="gpt-4o")
        output_instructions = []

        for template in templates:
            if template['template_type'] == "ORIENTATION":
                system_prompt = {
                                "role": "system",
                                "content": """
                                            You are a language assistant for a robot navigation system. Your task is to generate a single concise, natural language instruction in imperative English based on an ORIENTATION template.

                                            The template specifies:
                                            - An absolute direction (East, South, West, or North) the robot should face.
                                            - A visible landmark (building) and its relative position (left/right).

                                            Your instruction must explicitly follow this order:
                                            1. Clearly state the facing direction first:
                                                - "Face north."
                                                - "Turn to face east."
                                                - "Make sure you are facing south."
                                            2. Then briefly describe the landmark and its relative position using only essential visual features (maximum of one or two extra features):
                                                - Example good concise description:
                                                    - "You will see a gray commercial building with a large entrance awning on your right."
                                                    - "A tall glass office building will be on your left."
                                                    - "You should notice the red-brick store with rectangular windows on your right."

                                            Important:
                                            - DO NOT list all extra_features provided in the description. Pick at most one or two distinctive, easily visible ground-level features.
                                            - DO NOT include any movement instructions (e.g., "move forward").
                                            - Keep your sentence clear, concise, and fluent.

                                            Only return the final instruction.
                                            """
                                }
            elif template['template_type'] == "FINISH":
                system_prompt = {
                                "role": "system",
                                "content": """
                                You are a language assistant for a robot navigation system. Your task is to generate a single, fluent, and natural-sounding final navigation instruction in imperative English based on a FINISH template.

                                The template provides:
                                - A landmark description (building).
                                - The side (left or right) on which the building will appear.

                                Your instruction should clearly indicate:
                                - The robot must stop upon reaching the described building.
                                - The robot must face toward that building upon stopping.
                                - Mention explicitly the side (left/right) where the building will appear.
                                - Provide at most one or two clearly visible ground-level features of the building in a concise manner.

                                Important:
                                Vary your opening phrases naturally. Do NOT always start with "If you keep walking,". Use diverse opening phrases like:
                                - "Continue straight until you see..."
                                - "Keep going until you notice..."
                                - "Walk ahead, and you'll find..."
                                - "Proceed forward until reaching..."
                                - "Continue forward, and you will see..."

                                Example outputs (varied phrasing):
                                - "Continue straight until you see the gray commercial building with large glass windows on your right, stop and face that building."
                                - "Keep going until you notice the brown brick apartment with dragon signage on your left, stop and face that building."
                                - "Walk ahead, and you'll find the white office building with an entrance awning on your right, stop and face that building."

                                Only return the final instruction.
                                """
                                }
            elif template['template_type'] == "ROAD":
                if using_number_of_intersections:
                    system_prompt = {
                        "role": "system",
                        "content": """
                                You are a language assistant for a robot navigation system. Your task is to generate a single fluent, concise, and natural-sounding navigation instruction in imperative English based on a ROAD template.

                                The template provides:
                                - An intersection number (e.g., "2nd", "3rd").

                                Your instruction must:
                                - Direct the robot clearly to keep moving forward and stop upon reaching the specified intersection.
                                - Include the intersection number explicitly.
                                - NOT mention landmarks or buildings.
                                - NOT include any turning directions (left/right).
                                - Use natural, varied phrasing to enhance linguistic diversity.

                                Suggested diverse sentence structures:
                                - "Continue forward until you reach the [number] intersection, then stop."
                                - "Keep moving straight and halt when you arrive at the [number] intersection."
                                - "Proceed ahead, stopping at the [number] intersection."
                                - "Walk straight and come to a stop at the [number] intersection."
                                - "Move forward until reaching the [number] intersection, then stop there."

                                Examples:
                                - "Continue forward until you reach the second intersection, then stop."
                                - "Keep moving straight and halt when you arrive at the third intersection."
                                - "Proceed ahead, stopping at the fourth intersection."

                                Only return the final instruction.
                                """
                                }
                else:
                    system_prompt = {
                                "role": "system",
                                "content": """
                                You are a language assistant for a robot navigation system. Your task is to generate a single concise, fluent, and natural-sounding navigation instruction in imperative English based on a ROAD template.

                                The template provides:
                                - A landmark description (building) to clearly identify the intersection.

                                Your instruction must:
                                - Direct the robot to continue moving forward.
                                - Clearly instruct the robot to stop upon reaching the intersection identified by the landmark.
                                - Mention at most one or two easily visible ground-level features of the landmark building.
                                - NOT include any turning direction (left/right).
                                - NOT mention intersection numbers.

                                Examples:
                                - "Keep moving forward and stop at the intersection with the dark gray office building featuring large glass windows on your right."
                                - "Proceed straight and stop at the intersection where you see the red brick store on your left."

                                Only return the final instruction.
                                """
                                }
            elif template['template_type'] == "TURNING":
                system_prompt = {
                                "role": "system",
                                "content": """
                                You are a language assistant for a robot navigation system. Your task is to generate a single concise, clear, and natural-sounding navigation instruction in imperative English based on a TURNING template.

                                The template provides:
                                - A turning direction ("left" or "right").

                                Your instruction must:
                                - Explicitly mention that the turn occurs at an intersection.
                                - Clearly and briefly instruct the robot to turn in the specified direction.
                                - Use direct, imperative, and natural phrasing.

                                Examples:
                                - "Turn right at the intersection."
                                - "Make a left turn at the intersection."
                                - "At the intersection, take a right."

                                Only return the final instruction.
                                """
                                }
            else:
                raise ValueError(f"Invalid template type: {template['template_type']}")
            
            user_prompt_text = f"""Convert the following structured navigation template into a natural language instruction: {template}"""
            
            print(f"Generating instruction for template type: {template['template_type']}")
            response = model.chat(
                system_prompt=system_prompt["content"],
                user_prompt=user_prompt_text,
                image=None,
                max_tokens=512,
                temperature=0.1
            )
            print(f"Generated instruction: {response}")
            print("-" * 80)
            output_instructions.append(response)
        return output_instructions
    
    @staticmethod
    def calculate_orientation(node, next_node):
        orientation = (next_node['pos'] - node['pos']) / np.linalg.norm(next_node['pos'] - node['pos'])
        # Convert orientation vector to angle in degrees
        if np.allclose(orientation, [1, 0]):
            return 0, "West"
        elif np.allclose(orientation, [-1, 0]):
            return 180, "East"  
        elif np.allclose(orientation, [0, 1]):
            return 90, "North"
        elif np.allclose(orientation, [0, -1]): 
            return 270, "South"
        return np.degrees(np.arctan2(orientation[1], orientation[0]))
    
    def exit(self):
        self.unrealcv_client.client.disconnect()
