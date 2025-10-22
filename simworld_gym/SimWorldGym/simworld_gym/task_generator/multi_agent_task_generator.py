from typing import List
import os
import json
import re
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
from SimWorldGym.simworld_gym.task_generator.map_utils.utils.utils import visualize_map_multiagents_spawning_locations, visualize_map_with_all_landmarks

from collections import defaultdict

class MultiAgentTaskGenerator:
    def __init__(self, seed: int = 42, load_map: bool = True):
        """Initialize the route generator."""
        self.seed = seed
        random.seed(seed)
        self._init_unrealcv(load_map)

    def _load_map(self):
        """Load map from the template JSON file."""
        self.progen_city = json.load(open(self.city_path))["nodes"]
        self.city = DataImporter.import_city_data(self.city_path)
        self.cfg = CityFunctionCall(self.city_path, self.city)
        self.city_route_generator = RouteGenerator(RouteManager())
        self.map = self._load_strctured_map()

    def _init_unrealcv(self, if_load_map: bool = True):
        resolution = (720, 600)
        self.unrealcv_client = UnrealCV(port=9000, ip='127.0.0.1', resolution=resolution)
        # self.unrealcv_client.client.request("vset /cameras/spawn")
        if if_load_map:
            self.unrealcv_client.spawn_bp_asset("/Game/Human_Avatar/DefaultCharacter/Blueprint/BP_Default_Character.BP_Default_Character_C", "picture_man")
            self.unrealcv_client.spawn_bp_asset("/Game/Robot_Dog/Blueprint/BP_SpotRobot.BP_SpotRobot_C", "spot_robot")
            time.sleep(1)
            self.unrealcv_client.client.request("vset /camera/1/size {} {}".format(resolution[0], resolution[1]))
            self.unrealcv_client.client.request("vset /camera/1/reflection Lumen")
            self.unrealcv_client.client.request("vset /camera/1/illumination Lumen")
            self.unrealcv_client.client.request("vset /camera/1/exposure_bias 4.5")
            self.unrealcv_client.client.request("vset /camera/2/size {} {}".format(resolution[0], resolution[1]))
            self.unrealcv_client.client.request("vset /camera/2/reflection Lumen")
            self.unrealcv_client.client.request("vset /camera/2/illumination Lumen")
            self.unrealcv_client.client.request("vset /camera/2/exposure_bias 4.5")
            self.unrealcv_client.client.request("vset /camera/2/fov 120")

    
    def _load_buildings_on_street(self):
        # Group buildings by road_index using defaultdict for efficient classification
        self.buildings_by_road = defaultdict(list)
        for building in self.city.building_manager.buildings:
            self.buildings_by_road[building.road_index].append(building)
    
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
        map.interpolate_nodes(num_points=10)
        return map

    def load_env(self, city_path: str, road_path: str, if_load_map: bool = True):
        self.city_path = city_path
        self.road_path = road_path
        self.city = DataImporter.import_city_data(city_path)
        self.cfg = CityFunctionCall(city_path, self.city)
        self.route_generator = RouteGenerator(RouteManager())
        self.map = self._load_strctured_map()
        self._load_buildings_on_street()
        if if_load_map:
            self.env_generator = EnvironmentGenerator(self.unrealcv_client)
            self.env_generator.clear_env()
            self.env_generator.loading_json(city_path)
            self.env_generator.generate_world()
    
    def _calculate_front_door_position(self, building) -> Vector:
        """Calculate the front door position for a given asset."""
        road_index = int(re.search(r'GEN_Road_(\d+)', building.road_index).group(1))
        road = self.city.road_manager.roads[road_index]
        if abs(road.start.x - road.end.x) < 1e-3:
            if building.center.x > road.start.x:
                front_door_position = [road.start.x+17, building.center.y]
            else:
                front_door_position = [road.start.x-17, building.center.y]
        elif abs(road.start.y - road.end.y) < 1e-3:
            if building.center.y > road.start.y:
                front_door_position = [building.center.x, road.start.y+17]
            else:
                front_door_position = [building.center.x, road.start.y-17]
        else:
            raise ValueError(f"Road {road_index} is not a straight line")
        front_door_position = [front_door_position[0]*100, front_door_position[1]*100]
        return front_door_position
    
    def _calculate_camera_transformation(self, building) -> Vector:
        """Calculate the camera position for a given asset."""
        road_index = int(re.search(r'GEN_Road_(\d+)', building.road_index).group(1))   
        road = self.city.road_manager.roads[road_index]
        if abs(road.start.x - road.end.x) < 1e-3:
            if building.center.x > road.start.x:
                camera_position = [road.start.x-8, building.center.y]
                camera_orientation = [0, 0, 0]
            else:
                camera_position = [road.start.x+8, building.center.y]
                camera_orientation = [0, 0, 180]
        elif abs(road.start.y - road.end.y) < 1e-3:
            if building.center.y > road.start.y:
                camera_position = [building.center.x, road.start.y-8]
                camera_orientation = [0, 0, 90]
            else:
                camera_position = [building.center.x, road.start.y+8]
                camera_orientation = [0, 0, -90]
        else:
            raise ValueError(f"Road {road_index} is not a straight line")
        camera_position = [camera_position[0]*100, camera_position[1]*100]
        return camera_position, camera_orientation


    def _sample_buildings_on_street(self):
        # Count frequency of each building type
        building_counts = {}
        total_buildings = 0
        for road_buildings in self.buildings_by_road.values():
            for building in road_buildings:
                building_type = building.building_type.name
                building_counts[building_type] = building_counts.get(building_type, 0) + 1
                total_buildings += 1

        # Calculate weights inversely proportional to frequency
        building_weights = {}
        for building_type, count in building_counts.items():
            building_weights[building_type] = total_buildings / count

        buildings = {}
        for road, road_buildings in self.buildings_by_road.items():
            # Group buildings by type
            buildings_by_type = {}
            for building in road_buildings:
                building_type = building.building_type.name
                if building_type not in buildings_by_type:
                    buildings_by_type[building_type] = []
                buildings_by_type[building_type].append(building)
            
            # Sample from each type based on weights
            sampled_buildings = []
            building_types = list(buildings_by_type.keys())
            
            # Try to sample 3 different types if possible
            if len(building_types) >= 3:
                # Sample types based on their weights
                type_weights = [building_weights[bt] for bt in building_types]
                selected_types = []
                remaining_types = building_types.copy()
                remaining_weights = type_weights.copy()
                for _ in range(3):
                    chosen_idx = random.choices(range(len(remaining_types)), weights=remaining_weights, k=1)[0]
                    selected_types.append(remaining_types[chosen_idx])
                    remaining_types.pop(chosen_idx)
                    remaining_weights.pop(chosen_idx)
                for btype in selected_types:
                    type_buildings = buildings_by_type[btype]
                    selected_building = random.choice(type_buildings)
                    sampled_buildings.append({"building": selected_building, "camera_position": self._calculate_camera_transformation(selected_building)[0], "camera_orientation": self._calculate_camera_transformation(selected_building)[1], "front_door_position": self._calculate_front_door_position(selected_building)})
            else:
                # If less than 3 types, take one from each available type
                for btype in building_types:
                    type_buildings = buildings_by_type[btype]
                    weights = [building_weights[btype]] * len(type_buildings)
                    selected_building = random.choices(type_buildings, weights=weights, k=1)[0]
                    sampled_buildings.append({"building": selected_building, "camera_position": self._calculate_camera_transformation(selected_building)[0], "camera_orientation": self._calculate_camera_transformation(selected_building)[1], "front_door_position": self._calculate_front_door_position(selected_building)})
            buildings[road] = sampled_buildings
        return buildings
    
    def visualize_task(self, buildings: List[dict], save_path: Optional[str] = None):
        """
        Visualize the task on the map.
        """

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
            visualize_map_with_all_landmarks(self.map, buildings, save_path=save_path)
        else:
            visualize_map_with_all_landmarks(self.map, buildings)

    def _save_landmark_images(self, nodes_of_interest: List[Point], unrealcv_client: UnrealCV, save_path: str):
        """
        Save the landmark images.
        """
        landmark_images = []
        name_of_landmark_images = []
        for i, node in enumerate(nodes_of_interest):
            camera_orientation = node['camera_orientation']
            camera_position = node['camera_position']

            landmark_path = os.path.join(save_path, f"landmark_{i}.png")
            name_of_landmark_images.append(f"landmark_{i}.png")
            # unrealcv_client.client.request("vset /camera/0/location {} {} {}".format(camera_position.x, camera_position.y, 200))
            # unrealcv_client.client.request("vset /camera/0/rotation {} {} {}".format(0, camera_orientation[2], 0))
            # time.sleep(0.2)
            unrealcv_client.set_location([camera_position[0], camera_position[1], 92.150003], "picture_man")
            unrealcv_client.custom_set_orientation([camera_orientation[0], camera_orientation[1], camera_orientation[2]], "picture_man")
            unrealcv_client.client.request("vset /camera/1/location {} {} {}".format(camera_position[0], camera_position[1], 200))
            unrealcv_client.client.request("vset /camera/1/rotation {} {} {}".format(0, camera_orientation[2], 0))

            time.sleep(2)
            # unrealcv_client.read_image(1, "lit", "file_path", landmark_path)
            img = unrealcv_client.read_image(1, "lit", "direct")
            cv2.imwrite(landmark_path, img)
            landmark_images.append(img)
        return landmark_images, name_of_landmark_images
    
    def _save_robot_view_image(self, buildings: List[dict], unrealcv_client: UnrealCV, save_path: str):
        """
        Save the landmark images.
        """
        robot_view_images = []
        name_of_robot_view_images = []
        for i, building in enumerate(buildings):
            robot_spawning_orientation = building['camera_orientation']
            robot_spawning_position = building['front_door_position']

            robot_view_path = os.path.join(save_path, f"robot_view_{i}.png")
            name_of_robot_view_images.append(f"robot_view_{i}.png")
            # time.sleep(0.2)
            unrealcv_client.set_location([robot_spawning_position[0], robot_spawning_position[1], 67.818145], "spot_robot")
            unrealcv_client.custom_set_orientation([robot_spawning_orientation[0], robot_spawning_orientation[1], robot_spawning_orientation[2]], "spot_robot")
            time.sleep(2)
            # unrealcv_client.read_image(1, "lit", "file_path", landmark_path)
            img = unrealcv_client.read_image(2, "lit", "direct")
            cv2.imwrite(robot_view_path, img)
            robot_view_images.append(img)
        return robot_view_images, name_of_robot_view_images
    
    def generate_images(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self.buildings = self._sample_buildings_on_street()
        total_buildings = []
        for road, road_buildings in self.buildings.items():
            total_buildings.extend(road_buildings)
        self.total_buildings = total_buildings
        landmark_img_path = os.path.join(save_path, "landmark_images")
        os.makedirs(landmark_img_path, exist_ok=True)
        self._save_landmark_images(total_buildings, self.unrealcv_client, landmark_img_path)
        output_img_path = os.path.join(save_path, "map.png")
        visualize_map_with_all_landmarks(self.map, total_buildings, save_path=output_img_path)
        robot_view_img_path = os.path.join(save_path, "robot_view_images")
        os.makedirs(robot_view_img_path, exist_ok=True)
        self._save_robot_view_image(total_buildings, self.unrealcv_client, robot_view_img_path)

        i = 0
        road_buildings_json = {}
        for road, road_buildings in self.buildings.items():
            building_json = []
            for i, building in enumerate(road_buildings):
                building_infor = {"building_id": i, "building_position": building["front_door_position"], "building_orientation": building["camera_orientation"], "building_type": building['building'].building_type.name}
                building_json.append(building_infor)
            road_buildings_json[road] = building_json
        json.dump(road_buildings_json, open(os.path.join(save_path, "road_buildings.json"), "w"), indent=4)

        total_buildings_json = []
        for i, building in enumerate(self.total_buildings):
            building_infor = {"building_id": i, "building_position": building["front_door_position"], "building_orientation": building["camera_orientation"], "building_type": building['building'].building_type.name}
            total_buildings_json.append(building_infor)
        json.dump(total_buildings_json, open(os.path.join(save_path, "total_buildings.json"), "w"), indent=4)

    def _sample_spawning_locations(self, buildings: List[dict], mode: str = "easy"):
        """
        Sample the spawning locations for the agents.
        Args:
            buildings: List of building dictionaries
            mode: "easy" for same road, "hard" for different roads
        Returns:
            List of sampled buildings
        """
        # Get all available roads that have buildings
        available_roads = list(buildings.keys())
        if not available_roads:
            raise ValueError("No roads with buildings available")
            
        if mode == "easy":
            # Sample one road and get two buildings from it
            selected_road = random.choice(available_roads)
            road_buildings = buildings[selected_road]
            if len(road_buildings) < 2:
                raise ValueError(f"Road {selected_road} has less than 2 buildings")
            sampled_buildings = random.sample(road_buildings, 2)
            
        elif mode == "hard":
            # Sample two different roads and get one building from each
            if len(available_roads) < 2:
                raise ValueError("Not enough roads with buildings for hard mode")
            selected_roads = random.sample(available_roads, 2)
            sampled_buildings = []
            for road in selected_roads:
                road_buildings = buildings[road]
                if not road_buildings:
                    raise ValueError(f"Road {road} has no buildings")
                sampled_buildings.append(random.choice(road_buildings))
        
        spawning_locations = [building['building_position'] for building in sampled_buildings]
        spawning_orientations = [building['building_orientation'] for building in sampled_buildings]
        return spawning_locations, spawning_orientations, sampled_buildings[1]['building_id']

    def generate_task(self, map_path: str, city_path: str, road_path: str, sample_mode: str, save_path: str, set_index: int, dist_threshold: int):
        os.makedirs(save_path, exist_ok=True)
        map_file_path = os.path.join(map_path, "road_buildings.json")
        buildings = json.load(open(map_file_path))
        spawning_locations, spawning_orientations, agent2_spawning_landmark_index = self._sample_spawning_locations(buildings, sample_mode)
        # Calculate distance between spawning locations
        distance_between_agents = int(((spawning_locations[0][0] - spawning_locations[1][0]) ** 2 + 
                                 (spawning_locations[0][1] - spawning_locations[1][1]) ** 2) ** 0.5 / 100)
        max_steps = distance_between_agents * 10

        self.city_path = city_path
        self.road_path = road_path
        self.city = DataImporter.import_city_data(city_path)
        self.cfg = CityFunctionCall(city_path, self.city)
        self.route_generator = RouteGenerator(RouteManager())
        self.map = self._load_strctured_map()

        task_folder = os.path.join(save_path, f"task_dist_{distance_between_agents}_{set_index}_1")
        os.makedirs(task_folder, exist_ok=True)
        with open(city_path, 'r', encoding='utf-8') as f:
            progen_city = json.load(f)
        with open(os.path.join(task_folder, "progen_world.json"), 'w') as f:
            json.dump(progen_city, f, indent=2)

        with open(road_path, 'r', encoding='utf-8') as f:
            progen_road = json.load(f)
        with open(os.path.join(task_folder, "roads.json"), 'w') as f:
            json.dump(progen_road, f, indent=2)
        task_config = self._generate_task_config(os.path.basename(map_path),agent2_spawning_landmark_index,spawning_locations, spawning_orientations, max_steps, dist_threshold)
        with open(os.path.join(task_folder, "task_config.json"), 'w') as f:
            json.dump(task_config, f, indent=2)

        visualize_map_multiagents_spawning_locations(self.map, spawning_locations, save_path=os.path.join(task_folder, "map.png"))
       
    def _generate_task_config(self, map_path: str, agent2_spawning_landmark_index: int,spawning_locations: List[List[float]], spawning_orientations: List[float], max_steps: int, dist_threshold: int):
        """
        Generate a task config for an agent.
        """
        def _make_agent_config(idx, spawning_loc, spawning_orient):
            return {
                "type": "spot",
                "instance_name": "BP_SpotRobot_C", 
                "agent_name": f"Agent_{idx+1}",
                "radius": 70,
                "benchmark_type": 1,
                "camera_id": idx+1,
                "destination" : {
                    "x": -100,
                    "y": -100,
                    "z": -100
                },
                "spawning_point": {
                    "location": {
                        "x": spawning_loc[0],
                        "y": spawning_loc[1],
                        "z": 73.957685
                    },
                    "orientation": {
                        "roll": 0, "pitch": 0, "yaw": spawning_orient[2]
                    }
                }
            }

        return {
            "criteria": {
                "max_steps": max_steps,
                "dist_threshold": dist_threshold
            },
            "map_index": map_path,
            "agent2_spawning_landmark_index": agent2_spawning_landmark_index,
            "agents": [
                _make_agent_config(i, loc, orient) 
                for i, (loc, orient) in enumerate(zip(spawning_locations, spawning_orientations))
            ]
        }