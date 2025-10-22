from typing import List, Dict, Any
import numpy as np
import os
import json
import re
from dataclasses import dataclass
import random
from typing import Optional

import time
from simworld_gym.utils.unrealcv_basic import UnrealCV
from simworld_gym.utils.environment_generator import EnvironmentGenerator
import cv2

from simworld.citygen.route.route_generator import RouteGenerator
from simworld.citygen.function_call.city_function_call import CityFunctionCall
from simworld.citygen.route.route_manager import RouteManager
from simworld.citygen.dataclass.dataclass import Point
from simworld.utils.data_importer import DataImporter

import json
from simworld_gym import Config, Vector, Road, Node, Edge, Map
from simworld_gym.task_generator.map_utils.utils.utils import visualize_map_with_path

from llm_utils.customized_openai_model import CustomizedOpenAIModel

@staticmethod
def calculate_correct_robot_orientation_action(robot_current_orientation, correct_robot_orientation, tolerance=5):
    robot_current_orientation %= 360
    correct_robot_orientation %= 360

    delta = (correct_robot_orientation - robot_current_orientation) % 360

    if abs(delta) <= tolerance or abs(delta - 360) <= tolerance:
        return [-1]
    elif abs(delta - 90) <= tolerance:
        return [4, -1]
    elif abs(delta - 180) <= tolerance:
        return [4, 4, -1]
    elif abs(delta - 270) <= tolerance:
        return [5, -1]
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
        num_move_forwards = round((current_robot_location[0]-correct_robot_location[0])/500)
    elif current_robot_orientation == 270:
        num_move_forwards = round((current_robot_location[1]-correct_robot_location[1])/500)
    else:
        raise ValueError(f"Invalid orientation: {current_robot_orientation}")
    if num_move_forwards < 0:
        raise ValueError(f"Target location {correct_robot_location} cannot be reached by moving forwards from {current_robot_location} with orientation {current_robot_orientation}")
    current_robot_location = calculate_robot_location_after_move_forwards(current_robot_location, current_robot_orientation, num_move_forwards)
    return [0] * num_move_forwards + [-1], current_robot_location

@staticmethod
def calculate_correct_robot_turning_action(current_robot_location, current_robot_orientation, correct_robot_location, correct_robot_orientation, tolerance=5):
    action_before_turn, current_robot_location = calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, correct_robot_location)
    action_before_turn = action_before_turn[:-1]
    turn_action = calculate_correct_robot_orientation_action(current_robot_orientation, correct_robot_orientation, tolerance)[:-1]
    current_robot_orientation = correct_robot_orientation
    action_after_turn, current_robot_location = calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, correct_robot_location)
    output_action = action_before_turn + turn_action + action_after_turn
    return output_action, current_robot_location, current_robot_orientation

@staticmethod
def calculate_correct_robot_finish_action(current_robot_location, current_robot_orientation, correct_robot_location, correct_robot_orientation, tolerance=5):
    action_before_turn, current_robot_location = calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, correct_robot_location)
    action_before_turn = action_before_turn[:-1]
    turn_action = calculate_correct_robot_orientation_action(current_robot_orientation, correct_robot_orientation, tolerance)
    current_robot_orientation = correct_robot_orientation
    output_action = action_before_turn + turn_action
    return output_action, current_robot_location, current_robot_orientation

class TaskRouteGenerator:
    def __init__(self, seed: int = 42, street_library_path: str = None, if_load_map: bool = True):
        """Initialize the route generator."""
        self._init_unrealcv(if_load_map)
        self.seed = seed
        self.street_library_path = street_library_path
        self.static_description = self._load_static_description()
        random.seed(seed)
    
    def _init_unrealcv(self, if_load_map: bool = True):
        resolution = (720, 600)
        self.unrealcv_client = UnrealCV(port=9000, ip='127.0.0.1', resolution=resolution)
        if if_load_map:
            self.unrealcv_client.spawn_bp_asset("/Game/Robot_Dog/Blueprint/BP_SpotRobot.BP_SpotRobot_C", "spot_robot")
            resolution = (720, 600)
            time.sleep(1)
            self.unrealcv_client.client.request("vset /camera/1/size {} {}".format(resolution[0], resolution[1]))
            self.unrealcv_client.client.request("vset /camera/1/reflection Lumen")
            self.unrealcv_client.client.request("vset /camera/1/illumination Lumen")
            self.unrealcv_client.client.request("vset /camera/1/exposure_bias 4.5")
            self.unrealcv_client.client.request("vset /camera/1/fov 120")

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
        self.map = self._load_strctured_map()

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
        map.interpolate_nodes(num_points=-1)
        return map
    
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
    
    def load_env(self, city_path: str, road_path: str, if_load_map: bool = True):
        self.city_path = city_path
        self.road_path = road_path
        self._load_map()
        if if_load_map:
            self.env_generator = EnvironmentGenerator(self.unrealcv_client)
            self.env_generator.clear_env()
            self.env_generator.loading_json(city_path)
            self.env_generator.generate_world()
            self.unrealcv_client.clean_garbage()
            time.sleep(1)
    
    def _transform_path_to_nodes(self, path: List[Node]) -> List[Dict[str, Any]]:
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
        print("===============FINISH LOADING NODES===========")
        return nodes

    def _update_path_with_correct_spawning_and_finishing_location(self, nodes: List[Dict[str, Any]], nodes_of_interest: List[Dict[str, Any]]):
        """
        Update the nodes with the correct spawning and finishing condition.
        """
        spawning_location = [nodes_of_interest[0]['robot_location'][0], nodes_of_interest[0]['robot_location'][1]]
        finishing_location = [nodes_of_interest[-1]['robot_location'][0], nodes_of_interest[-1]['robot_location'][1]]
        print('nodes_of_interest:', nodes_of_interest)
        print('spawning_location:', spawning_location)

        direction = (nodes[1]['pos'] - nodes[0]['pos'])/np.linalg.norm(nodes[1]['pos'] - nodes[0]['pos'])
        updated_nodes = nodes.copy()
        
        # Check if direction is [0,-1] (south)
        if abs(direction[0]) < 1e-6 and abs(direction[1] + 1) < 1e-6:
            while True:
                if spawning_location[1] > updated_nodes[0]['pos'][1]:
                    updated_nodes.insert(0, {'pos': np.array(spawning_location), 'type': 'normal', 'road': updated_nodes[0]['road']})
                    break
                else:
                    updated_nodes.pop(0)
        # Check if direction is [0,1] (north)
        elif abs(direction[0]) < 1e-6 and abs(direction[1] - 1) < 1e-6:
            while True:
                if spawning_location[1] < updated_nodes[0]['pos'][1]:
                    updated_nodes.insert(0, {'pos': np.array(spawning_location), 'type': 'normal', 'road': updated_nodes[0]['road']})
                    break
                else:
                    updated_nodes.pop(0)
        # Check if direction is [-1,0] (west)
        elif abs(direction[0] + 1) < 1e-6 and abs(direction[1]) < 1e-6:
            while True:
                if spawning_location[0] > updated_nodes[0]['pos'][0]:
                    updated_nodes.insert(0, {'pos': np.array(spawning_location), 'type': 'normal', 'road': updated_nodes[0]['road']})
                    break
                else:
                    updated_nodes.pop(0)
        # Check if direction is [1,0] (east)
        elif abs(direction[0] - 1) < 1e-6 and abs(direction[1]) < 1e-6:
            while True:
                if spawning_location[0] < updated_nodes[0]['pos'][0]:
                    updated_nodes.insert(0, {'pos': np.array(spawning_location), 'type': 'normal', 'road': updated_nodes[0]['road']})
                    break
                else:
                    updated_nodes.pop(0)

        end_direction = (nodes[-1]['pos'] - nodes[-2]['pos'])/np.linalg.norm(nodes[-1]['pos'] - nodes[-2]['pos'])
        # Check if end_direction is [0,-1] (south)
        if abs(end_direction[0]) < 1e-6 and abs(end_direction[1] + 1) < 1e-6:
            while True:
                if finishing_location[1] < updated_nodes[-1]['pos'][1]:
                    updated_nodes.append({'pos': np.array(finishing_location), 'type': 'normal', 'road': updated_nodes[-1]['road']})
                    break
                else:
                    updated_nodes.pop(-1)
        # Check if end_direction is [0,1] (north)
        elif abs(end_direction[0]) < 1e-6 and abs(end_direction[1] - 1) < 1e-6:
            while True:
                if finishing_location[1] > updated_nodes[-1]['pos'][1]:
                    updated_nodes.append({'pos': np.array(finishing_location), 'type': 'normal', 'road': updated_nodes[-1]['road']})
                    break
                else:
                    updated_nodes.pop(-1)
        # Check if end_direction is [-1,0] (west)
        elif abs(end_direction[0] + 1) < 1e-6 and abs(end_direction[1]) < 1e-6:
            while True:
                if finishing_location[0] < updated_nodes[-1]['pos'][0]:
                    updated_nodes.append({'pos': np.array(finishing_location), 'type': 'normal', 'road': updated_nodes[-1]['road']})
                    break
                else:
                    updated_nodes.pop(-1)
        # Check if end_direction is [1,0] (east)
        elif abs(end_direction[0] - 1) < 1e-6 and abs(end_direction[1]) < 1e-6:
            while True:
                if finishing_location[0] > updated_nodes[-1]['pos'][0]:
                    updated_nodes.append({'pos': np.array(finishing_location), 'type': 'normal', 'road': updated_nodes[-1]['road']})
                    break
                else:
                    updated_nodes.pop(-1)

        return updated_nodes
    
    
    def generate_task_with_less_turns(self, path: List[Node], distance: float, agent_config: Dict[str, Any], output_path: str, if_count_intersections: bool = True, set_index: int = 0) -> str:
        """
        Generate a task from the start and end points.
        """
        
        # Create output folder if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Create subfolder for each end node
        task_folder = os.path.join(output_path, f"task_dist_{len(path)}_{set_index}_1")

        os.makedirs(task_folder, exist_ok=True)
        nodes = self._transform_path_to_nodes(path)

        templates, nodes_of_interest, evaluation_of_subtask = self.nodes_to_templates(nodes, if_count_intersections)
        # return templates, nodes_of_interest, evaluation_of_subtask, task_folder, nodes
        nodes = self._update_path_with_correct_spawning_and_finishing_location(nodes, nodes_of_interest)

        description = self.generate_description_clear_finishing_condition(templates, if_count_intersections)
        # description = ""

        output_img_path = os.path.join(task_folder, "map.png")
        self.visualize_task(nodes, nodes_of_interest, output_img_path)
        print("Generated templates")

        spawning_orientation = random.choice([0, 90, 180, 270])

        landmark_images, intersection_images, name_of_landmark_images, name_of_robot_view_images = self._save_images(nodes_of_interest, self.unrealcv_client, task_folder)

        ground_truth_path = np.array([[node['pos'][0] * 100, node['pos'][1] * 100] for node in nodes])
        task_config = self._generate_task_config(spawning_location=nodes_of_interest[0]['robot_location'], 
                                                    spawning_orientation=spawning_orientation, 
                                                    destination=nodes_of_interest[-1]['robot_location'], 
                                                    instruction=description, 
                                                    max_steps=agent_config['distance_per_step'] * distance, 
                                                    dist_threshold=agent_config['dist_threshold'], 
                                                    ground_truth_path=ground_truth_path, 
                                                    name_of_landmark_images=name_of_landmark_images,
                                                    name_of_robot_view_images=name_of_robot_view_images,
                                                    evaluation_of_subtask=evaluation_of_subtask)
        
        
        output_config_path = os.path.join(task_folder, "task_config.json")
        with open(output_config_path, 'w') as f:
            json.dump(task_config, f, indent=2)

        with open(self.city_path, 'r', encoding='utf-8') as f:
            progen_city = json.load(f)
        with open(os.path.join(task_folder, "progen_world.json"), 'w') as f:
            json.dump(progen_city, f, indent=2)
        
        with open(self.road_path, 'r', encoding='utf-8') as f:
            progen_road = json.load(f)
        with open(os.path.join(task_folder, "roads.json"), 'w') as f:
            json.dump(progen_road, f, indent=2)
        
        print("Generated task config")
        # return templates, nodes_of_interest, evaluation_of_subtask, nodes

    def generate_training_task(self, map_path, task_name, output_path) -> str:
        """
        Generate a training task from the start and end points.
        """
        task_path = os.path.join(map_path, task_name)
        task_infor = json.load(open(os.path.join(task_path, "task_config.json")))
        spawning_orientation = task_infor['spawning_point']['orientation']['yaw']
        spawning_location = [task_infor['spawning_point']['location']['x'], task_infor['spawning_point']['location']['y']]
        instruction = task_infor['instruction']
        visual_instruction_images = [cv2.imread(os.path.join(task_path, image_name)) for image_name in task_infor['robot_view_images']]
        evaluation_of_subtask = task_infor['evaluation_of_subtask']

        # Generate ground truth actions in all the subtasks
        current_robot_orientation = spawning_orientation % 360
        current_robot_location = spawning_location
        actions = {}
        for i, evaluation in enumerate(evaluation_of_subtask):
            if evaluation['type'] == "ORIENTATION":
                subtask_actions = calculate_correct_robot_orientation_action(current_robot_orientation, evaluation['correct_robot_orientation'])
                current_robot_orientation = evaluation['correct_robot_orientation'] % 360
            elif evaluation['type'] == "ROAD":
                subtask_actions, current_robot_location = calculate_correct_robot_road_action(current_robot_location, current_robot_orientation, evaluation['correct_robot_location'])
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
            actions[i] = subtask_actions

        # Generate training examples at each step
        print("Starting to generate training examples...")
        current_robot_orientation = spawning_orientation % 360
        current_robot_location = spawning_location
        print(f"Initial robot orientation: {current_robot_orientation}")
        print(f"Initial robot location: {current_robot_location}")
        
        self.unrealcv_client.set_location_hard([current_robot_location[0], current_robot_location[1], 73.957685], "spot_robot")
        self.unrealcv_client.custom_set_orientation([0, 0, current_robot_orientation], "spot_robot")

        step = 0
        training_task_path = os.path.join(output_path, task_name)
        os.makedirs(training_task_path, exist_ok=True)
        print(f"Created training task directory at: {training_task_path}")
        
        for i, evaluation in enumerate(evaluation_of_subtask):
            print(f"\nProcessing subtask {i} of type: {evaluation['type']}")
            past_actions = []
            all_subtask_actions = actions[i]
            print(f"Number of actions in this subtask: {len(all_subtask_actions)}")
            
            for j, subtask_action in enumerate(all_subtask_actions):
                print(f"\nProcessing step {step} (action {j+1}/{len(all_subtask_actions)})")
                current_step_path = os.path.join(training_task_path, f"training_step_{step}_{evaluation['type']}")
                os.makedirs(current_step_path, exist_ok=True)
                print(f"Created step directory at: {current_step_path}")
                
                current_observation = self.unrealcv_client.read_image(1, "lit")
                print("Captured current observation image")
                
                # Calculate ground truth action based on evaluation type
                if subtask_action != -1:
                    print(f"Executing robot action: {subtask_action}")
                    current_robot_orientation, current_robot_location = self.robot_perform_action(subtask_action, current_robot_orientation, current_robot_location)
                
                current_instruction = instruction[i]
                current_visual_image = visual_instruction_images[i]
                current_direction = self.map_orientation_angle_to_direction(current_robot_orientation)
                if evaluation['type'] == "ORIENTATION":
                    distance2target = 0
                else:
                    distance2target = self.calculate_distance(current_robot_location, evaluation['correct_robot_location'])
                
                if evaluation['type'] == "ROAD":
                    target_direction = current_direction
                else:
                    target_direction = self.map_orientation_angle_to_direction(evaluation['correct_robot_orientation'])
                print(f"Current direction: {current_direction}")
                print(f"Target direction: {target_direction}")
                print(f"Distance to target: {distance2target}")
                
                ground_truth_action = all_subtask_actions[j:]
                self.write_current_step_data(current_observation, current_step_path, current_instruction, current_visual_image, past_actions, current_direction, target_direction, distance2target, ground_truth_action)
                print("Wrote step data to files")
                
                past_actions.append(all_subtask_actions[j])
                step += 1
        
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
                    node_of_interest[0]['robot_location'] = current_node['pos'] - (current_node['pos'] - nodes[node_index - 1]['pos']) / np.linalg.norm(current_node['pos'] - nodes[node_index - 1]['pos']) * 5
                    # node_of_interest[0]['robot_location'] = current_node['pos']
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
                    evaluation_of_subtask.append({"type": "TURNING", "correct_robot_orientation": current_robot_orientation, "correct_robot_position_area": correct_robot_position_area, "correct_robot_location": [nodes[node_index+num_nodes_between-1]['pos'][0]*100, nodes[node_index+num_nodes_between-1]['pos'][1]*100]})
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
        """
        pre_road_node = nodes[current_node_index - 1]
        # find the next road node
        next_road_index = None
        for i in range(current_node_index + 1, len(nodes)):
            if nodes[i]['type'] == 'normal':
                next_road_index = i
                break
        next_road_node = nodes[next_road_index]
        direction = self._calculate_cross_product(pre_node=pre_road_node, current_node=nodes[next_road_index-1], next_node=next_road_node)
        num_nodes_between = next_road_index - current_node_index
        return direction, num_nodes_between
       
    def visualize_task(self, nodes: List[Dict[str, Any]], nodes_of_interest: List[Point], save_path: Optional[str] = None):
        """
        Visualize the task on the map.
        """
        nodes_of_interest = [Point(node['position'].x*100, node['position'].y*100) for node in nodes_of_interest if 'position' in node]
        path = [Node(Vector(node['pos'][0]*100, node['pos'][1]*100), node['type']) for node in nodes]

        # roads = self.city.road_manager.roads
        
        # output_roads = [
        #     {
        #         'start': Vector(road.start.x * 100, road.start.y * 100),
        #         'end': Vector(road.end.x * 100, road.end.y * 100),
        #         'name': self.progen_city[i]['properties']['road_name']
        #     }
        #     for i, road in enumerate(roads)
        # ]
        
        if save_path:
            visualize_map_with_path(self.map, path, nodes_of_interest, save_path=save_path)
        else:
            visualize_map_with_path(self.map, path, nodes_of_interest)

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
    
    def _generate_task_config(self, spawning_location: np.ndarray, spawning_orientation: float, destination: np.ndarray, instruction: str, max_steps: int, dist_threshold: int, ground_truth_path: np.ndarray, name_of_landmark_images: str, name_of_robot_view_images: str, evaluation_of_subtask: List[Dict[str, Any]]):
        """
        Generate a task config for an agent.
        """
        return {
                "type" : "spot",
                "camera_id" : 1,
                "instance_name" : "BP_SpotRobot_C",
                "agent_name" : "SportRobot_1",
                "env_json_path": self.city_path,
                "radius" : 70,
                "benchmark_type" : 0,
                "destination" : {
                    "x": destination[0]*100,
                    "y": destination[1]*100,
                    "z": 73.957685
                },
                "criteria": {
                    "max_steps" : max_steps,
                    "dist_threshold" : dist_threshold
                },
                "spawning_point" : {
                    "location" : {
                    "x": spawning_location[0]*100,
                    "y": spawning_location[1]*100,
                    "z": 73.957685
                    },
                    "orientation" : {
                    "roll" : 0,
                    "pitch" : 0,
                    "yaw" : spawning_orientation
                    }
                },
                "instruction": instruction,
                "visual_path": "map.png",
                "landmark_images": name_of_landmark_images,
                "robot_view_images": name_of_robot_view_images,
                "ideal_path": ground_truth_path.tolist(),
                "evaluation_of_subtask": evaluation_of_subtask
                }
    
    def _save_images(self, nodes_of_interest: List[Point], unrealcv_client: UnrealCV, save_path: str):
        """
        Save the landmark and intersection images at the intersection area.
        """
        landmark_images = []
        intersection_images = []
        name_of_landmark_images = []
        name_of_robot_view_images = []
        unrealcv_client.clean_garbage()
        for i, node in enumerate(nodes_of_interest):
            # camera_orientation = node['camera_orientation']
            # camera_position = node['camera_position']

            # landmark_path = os.path.join(save_path, f"landmark_{i}.png")
            # name_of_landmark_images.append(f"landmark_{i}.png")
            # unrealcv_client.client.request("vset /camera/1/location {} {} {}".format(camera_position.x * 100, camera_position.y * 100, 200))
            # unrealcv_client.client.request("vset /camera/1/rotation {} {} {}".format(0, camera_orientation[2], 0))
            # time.sleep(1)
            # # unrealcv_client.read_image(1, "lit", "file_path", landmark_path)
            # img = unrealcv_client.read_image(1, "lit", "direct")
            # cv2.imwrite(landmark_path, img)
            # landmark_images.append(img)

            intersection_path = os.path.join(save_path, f"robot_view_{i}.png")
            name_of_robot_view_images.append(f"robot_view_{i}.png")

            unrealcv_client.set_location([node['robot_location'][0] * 100, node['robot_location'][1] * 100, 73.957685], "spot_robot")
            unrealcv_client.custom_set_orientation([0, 0, node['robot_orientation'][0]], "spot_robot")
            time.sleep(2)
            # unrealcv_client.read_image(1, "lit", "file_path", landmark_path)
            img = unrealcv_client.read_image(1, "lit", "direct")
            cv2.imwrite(intersection_path, img)
            intersection_images.append(img)

        return landmark_images, intersection_images, name_of_landmark_images, name_of_robot_view_images
    
    def generate_description_clear_finishing_condition(self, templates: List[str], using_number_of_intersections: bool = True) -> str:
        """
        Generate a natural language description from templates.
        
        Args:
            templates: List of template strings
            
        Returns:
            Natural language description
        """
        model = CustomizedOpenAIModel("gpt-4o")
        messages_batch = []

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
                    """,
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
                    """,
                }
            elif template['template_type'] == "ROAD":
                if using_number_of_intersections:
                    system_prompt = {
                        "role": "system",
                        "content": """
                            You are a language assistant for a robot navigation system. Your task is to generate a single concise, fluent, and natural-sounding navigation instruction in imperative English based on a ROAD template.

                            The template provides:
                            - An intersection number (e.g., "2nd", "3rd").

                            Your instruction must:
                            - Direct the robot to keep moving forward.
                            - Clearly instruct the robot to stop upon reaching the specified intersection number.
                            - NOT include any turning direction (left/right).
                            - NOT mention landmarks or buildings.
                            - Be concise, direct, and action-oriented.

                            Examples:
                            - "Continue forward until you reach the second intersection, then stop."
                            - "Keep going straight and stop at the third intersection."

                            Only return the final instruction.
                        """,
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
                            - NOT include any turning direction (left/right).
                            - NOT mention intersection numbers.

                            Examples:
                            - "Keep moving forward and stop at the intersection with the dark gray office building featuring large glass windows on your right."
                            - "Proceed straight and stop at the intersection where you see the red brick store on your left."

                            Only return the final instruction.
                        """,
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
                    """,
                }
            else:
                raise ValueError(f"Invalid template type: {template['template_type']}")

            user_prompt = {
                "role": "user",
                "content": f"Convert the following structured navigation template into a natural language instruction: {template}",
            }
            messages_batch.append([system_prompt, user_prompt])

        try:
            import asyncio as _asyncio
            try:
                _loop = _asyncio.get_running_loop()
                running = True
            except RuntimeError:
                running = False

            if running:
                responses = [model.generate(msgs) for msgs in messages_batch]
            else:
                async def _run():
                    return await model.generate_async(messages_batch)
                responses = _asyncio.run(_run())
        except Exception:
            responses = [model.generate(msgs) for msgs in messages_batch]

        output_instructions = responses
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
    
    @staticmethod
    def calculate_distance(current_robot_location, correct_robot_location):
        return np.linalg.norm(np.array(current_robot_location) - np.array(correct_robot_location))
    
    @staticmethod
    def map_orientation_angle_to_direction(orientation_angle):
        if orientation_angle == 0:
            return "West"
        elif orientation_angle == 90:
            return "North"
        elif orientation_angle == 180:
            return "East"
        elif orientation_angle == 270:
            return "South"
        else:
            raise ValueError(f"Invalid orientation angle: {orientation_angle}")
    
    def robot_perform_action(self,action, current_robot_orientation, current_robot_location):
        current_robot_orientation = current_robot_orientation % 360
        robot_name = 'spot_robot'
        speed = 5000
        duration = 0.1
        if action == 0:
            self.unrealcv_client.apply_action_transition(robot_name, [speed, duration, 0])
            current_robot_location = calculate_robot_location_after_move_forwards(current_robot_location, current_robot_orientation, 1)
        elif action == 4:
            self.unrealcv_client.apply_action_rotation(robot_name, [duration, 90, 1])
            current_robot_orientation = (current_robot_orientation + 90) % 360
        elif action == 5:
            self.unrealcv_client.apply_action_rotation(robot_name, [duration, -90, -1])
            current_robot_orientation = (current_robot_orientation - 90) % 360
        time.sleep(duration+0.05)
        # Wait until the action in Unreal Engine has finished executing
        is_available = self.unrealcv_client.get_is_available(robot_name)
        while not is_available:
            is_available = self.unrealcv_client.get_is_available(robot_name)
            time.sleep(0.05)
        return current_robot_orientation, current_robot_location

    @staticmethod
    def write_current_step_data(image, current_step_path, current_instruction, current_visual_image, past_actions, current_orientation, target_orientation, distance2target, ground_truth_action):
        cv2.imwrite(os.path.join(current_step_path, f"current_observation.png"), image)
        cv2.imwrite(os.path.join(current_step_path, f"visual_instruction.png"), current_visual_image)
        data = {
                    "text_instruction": current_instruction,
                    "visual_instruction": f"visual_instruction.png",
                    "past_actions": past_actions,
                    "current_observation": "current_observation.png",
                    "current_orientation": current_orientation,
                    "target_orientation": target_orientation,
                    "ground_truth_action": ground_truth_action,
                    "distance2target": distance2target
                }
        with open(os.path.join(current_step_path, f"meta_data.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    def exit(self):
        self.unrealcv_client.client.disconnect()
