import json
import os
import random
from typing import Dict, List, Optional
from llm_utils.customized_openai_model import CustomizedOpenAIModel
from simworld_gym.utils.environment_generator import EnvironmentGenerator

import numpy as np
import osmnx as ox
from simworld.citygen.dataclass.dataclass import Point
from simworld.citygen.function_call.city_function_call import CityFunctionCall
from simworld.utils.data_importer import DataImporter
from simworld.citygen.route.route_manager import RouteManager
from simworld.citygen.route.route_generator import RouteGenerator

import json
from simworld_gym import Config, Node, Edge, Map, Vector, Road
from simworld_gym.task_generator.map_utils.utils.utils import visualize_map_with_multiagents

from simworld_gym.utils.unrealcv_basic import UnrealCV
import cv2

import time



class DestinationGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.roads = {}
        self.assets = []
        self.unique_assets = []
        self.assets_template = {}
        self.static_templates = {}
        self.surrounding_templates = {}
        random.seed(seed)
    
    def _init_unrealcv(self, if_load_map: bool = True):
        resolution = (720, 600)
        self.unrealcv_client = UnrealCV(port=9000, ip='127.0.0.1', resolution=resolution)
        # self.unrealcv_client.client.request("vset /cameras/spawn")
        if if_load_map:
            self.unrealcv_client.spawn_bp_asset("/Game/Human_Avatar/DefaultCharacter/Blueprint/BP_Default_Character.BP_Default_Character_C", "picture_man")
            time.sleep(1)
            self.unrealcv_client.client.request("vset /camera/1/size {} {}".format(resolution[0], resolution[1]))
            self.unrealcv_client.client.request("vset /camera/1/reflection Lumen")
            self.unrealcv_client.client.request("vset /camera/1/illumination Lumen")
            self.unrealcv_client.client.request("vset /camera/1/exposure_bias 4.5")

    
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

        nodes_on_road = [[]] * len(road_objects)
        # Initialize the map
        for i, road in enumerate(road_objects):
            normal_vector = Vector(road.direction.y, -road.direction.x)
            point1 = road.start - normal_vector * (Config.SIDEWALK_OFFSET) + road.direction * Config.SIDEWALK_OFFSET
            point2 = road.end - normal_vector * (Config.SIDEWALK_OFFSET) - road.direction * Config.SIDEWALK_OFFSET

            point3 = road.end + normal_vector * (Config.SIDEWALK_OFFSET) - road.direction * Config.SIDEWALK_OFFSET
            point4 = road.start + normal_vector * (Config.SIDEWALK_OFFSET) + road.direction * Config.SIDEWALK_OFFSET

            node1 = Node(point1, "intersection")
            node2 = Node(point2, "intersection")
            node3 = Node(point3, "intersection")
            node4 = Node(point4, "intersection")

            nodes_on_road[i] = [node1, node2, node3, node4]

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
        map.interpolate_nodes(10)
        return map, nodes_on_road
    
    def load_env(self, city_path: str, road_path: str, mode: str = 'buildings', if_load_map: bool = True):
        self.city_path = city_path
        self.road_path = road_path
        self.city = DataImporter.import_city_data(city_path)
        self.cfg = CityFunctionCall(city_path, self.city)
        self.route_generator = RouteGenerator(RouteManager())
        self.map, self.nodes_on_road = self._load_strctured_map()
        self._load_assets(mode)
        self._load_template()
        self._init_unrealcv(if_load_map)
        if if_load_map:
            self.env_generator = EnvironmentGenerator(self.unrealcv_client)
            self.env_generator.clear_env()
            self.env_generator.loading_json(city_path)
            self.env_generator.generate_world()
    
    
    def _load_assets_static_template(self) -> Dict:
        """Load and filter static building descriptions from the template JSON file.
        
        Returns:
            Dict: Filtered dictionary mapping asset names to their static descriptions.
            Empty dict if file not found or invalid.
        """
        json_path = os.path.join(os.path.dirname(__file__), 'data', 'BuildingsDescriptionMapTemplate.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                template_library = json.load(f)
            return template_library
            
        except FileNotFoundError:
            print(f"Error: Template file not found at {json_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in template file at {json_path}")
            return {}

    def _load_assets_surrounding_template(self) -> Dict:
        """Generate templates describing each asset's surrounding environment.
        
        Analyzes the city layout to describe what's around each asset instance
        (nearby roads, buildings, etc).
        
        Returns:
            Dict: Mapping of asset instance IDs to their surrounding context.
        """
        surrounding_templates = {}

        # For each asset instance in the city
        for asset_id, asset_info in self.assets.items():
            # Get nearby buildings, roads, etc from city data
            nearby = self.route_generator._get_closest_building_object(asset_info["center"], self.city.city_quadtrees)
            road_id = self.cfg.find_building_segment_id(asset_id)
            # nearby_dict = {
            #     'roads': self.roads.get(road_id)
            # }
            nearby_dict = {}

            if nearby[1] is not None:  # Check if building info exists
                nearby_dict['buildings'] = {
                    'type': self.static_templates.get(nearby[1][0], {}).get('type', ''),
                    'color': self.static_templates.get(nearby[1][0], {}).get('color', ''), 
                    'material': self.static_templates.get(nearby[1][0], {}).get('material', ''),
                    'extra_features': self.static_templates.get(nearby[1][0], {}).get('extra_features', ''),
                    'distance': nearby[1][1]
                }

            # if nearby[0] is not None:  # Check if detail info exists
            #     nearby_dict['details'] = {
            #         'type': self.static_templates.get(nearby[0][0], {}).get('Type', ''),
            #         'distance': nearby[0][1]
            #     }
            
            # Store the surrounding context
            surrounding_templates[asset_id] = nearby_dict
            
        return surrounding_templates
    
    def _load_template(self) -> None:
        """Load and combine static and dynamic templates for all assets.
        
        Merges the static building descriptions with their surrounding context
        to create complete templates for each asset insance.
        """
        self.static_templates = self._load_assets_static_template()
        self.surrounding_templates = self._load_assets_surrounding_template()
        
        # Create complete template for each asset instance
        for asset_id, asset_info in self.assets.items():
            self.assets_template[asset_id] = {
                'asset_type': asset_info["instance_name"],
                'static_description': self.static_templates.get(asset_info["instance_name"], {}),
                'surrounding_context': self.surrounding_templates.get(asset_id, {})
            }

    def _load_assets(self, mode: str = 'buildings') -> None:
        """Load assets from the city data."""
        if mode.lower() not in ['buildings', 'details', 'all']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: 'buildings', 'details', 'all'")
        
        nodes = json.load(open(self.city_path))["nodes"]
        num_roads = len(self.city.road_manager.roads)
        self.roads = {nodes[i]["id"]: nodes[i]["properties"]["road_name"] for i in range(num_roads)}

        num_buildings = len(self.city.building_manager.buildings)
        num_details = len(self.city.detail_manager.details)
        # Use sets for efficient uniqueness checking and union operations
        if mode.lower() == 'buildings':
            self.assets = {nodes[i+num_roads]["id"] : {"instance_name": nodes[i+num_roads]["instance_name"], "center": self.city.building_manager.buildings[i].center} for i in range(num_buildings)}
            for asset_id, asset_info in self.assets.items():
                asset_info["camera_position"], asset_info["camera_orientation"] = self._calculate_camera_position(asset_id)
                asset_info["front_door_position"] = self._calculate_front_door_position(asset_id)
        elif mode.lower() == 'details':
            self.assets = {nodes[i+num_roads+num_buildings]["id"] : {"instance_name": nodes[i+num_roads+num_buildings]["instance_name"], "center": self.city.detail_manager.details[i].center} for i in range(num_details)}
        elif mode.lower() == 'all':
            self.assets = {nodes[i+num_roads]["id"] : {"instance_name": nodes[i+num_roads]["instance_name"], "center": self.city.building_manager.buildings[i].center} for i in range(num_buildings)}
            self.assets = {nodes[i+num_roads+num_buildings]["id"] : {"instance_name": nodes[i+num_roads+num_buildings]["instance_name"], "center": self.city.detail_manager.details[i].center} for i in range(num_details)}
        self.unique_assets = list(set(asset["instance_name"] for asset in self.assets.values()))
    
    def _calculate_front_door_position(self, asset_id: str) -> Vector:
        """Calculate the front door position for a given asset."""
        import re
        building_info = self.assets[asset_id]
        road_id = self.cfg.find_building_segment_id(asset_id)
        road_index = int(re.search(r'GEN_Road_(\d+)', road_id).group(1))
        road = self.city.road_manager.roads[road_index]
        if abs(road.start.x - road.end.x) < 1e-3:
            if building_info["center"].x > road.start.x:
                front_door_position = [road.start.x+17, building_info["center"].y]
            else:
                front_door_position = [road.start.x-17, building_info["center"].y]
        elif abs(road.start.y - road.end.y) < 1e-3:
            if building_info["center"].y > road.start.y:
                front_door_position = [building_info["center"].x, road.start.y+17]
            else:
                front_door_position = [building_info["center"].x, road.start.y-17]
        else:
            raise ValueError(f"Road {road_id} is not a straight line")
        front_door_position = [front_door_position[0]*100, front_door_position[1]*100]
        return front_door_position
    
    
    def _calculate_camera_position(self, asset_id: str) -> Vector:
        """Calculate the camera position for a given asset."""
        import re
        building_info = self.assets[asset_id]
        road_id = self.cfg.find_building_segment_id(asset_id)
        road_index = int(re.search(r'GEN_Road_(\d+)', road_id).group(1))
        road = self.city.road_manager.roads[road_index]
        if abs(road.start.x - road.end.x) < 1e-3:
            if building_info["center"].x > road.start.x:
                camera_position = [road.start.x-12, building_info["center"].y]
                camera_orientation = [0, 0, 0]
            else:
                camera_position = [road.start.x+12, building_info["center"].y]
                camera_orientation = [0, 0, 180]
        elif abs(road.start.y - road.end.y) < 1e-3:
            if building_info["center"].y > road.start.y:
                camera_position = [building_info["center"].x, road.start.y-12]
                camera_orientation = [0, 0, 90]
            else:
                camera_position = [building_info["center"].x, road.start.y+12]
                camera_orientation = [0, 0, -90]
        else:
            raise ValueError(f"Road {road_id} is not a straight line")
        camera_position = [camera_position[0]*100, camera_position[1]*100]
        return camera_position, camera_orientation
    
    def _filter_dict_values(self, source_dict: Dict, num_keys: int) -> Dict:
        """Filter dictionary values by randomly selecting keys and handling list values.
        
        Args:
            source_dict: Source dictionary to filter
            num_keys: Number of keys to randomly select
            
        Returns:
            Dict: Filtered dictionary with randomly selected keys and list values
        """
        filtered_dict = {}
        if not isinstance(source_dict, dict):
            return filtered_dict
            
        # Randomly select keys
        selected_keys = random.sample(list(source_dict.keys()), min(num_keys, len(source_dict)))
        
        # Handle each selected key-value pair
        for key in selected_keys:
            value = source_dict[key]
            if isinstance(value, list):
                # For list values, randomly select between 1/3 to 2/3 of the elements
                min_items = max(1, len(value) // 3)
                max_items = max(min_items, len(value) * 2 // 3)
                num_items = random.randint(min_items, max_items)
                filtered_dict[key] = random.sample(value, min(num_items, len(value)))
            else:
                # For non-list values, keep them as is
                filtered_dict[key] = value
                
        return filtered_dict

    def _count_exact_template_matches(self, generated_static: Dict, generated_surrounding: Dict) -> int:
        """Count how many assets in the original data exactly match the generated template.
        
        Args:
            generated_static: The generated static description
            generated_surrounding: The generated surrounding context
            
        Returns:
            int: Number of assets that exactly match the generated template
        """
        match_count = 0
        
        for asset_id, template in self.assets_template.items():
            # Check if static description matches exactly
            static_match = True
            for key, value in generated_static.items():
                template_value = template['static_description'].get(key)
                if isinstance(value, list):
                    # For list values, check if all elements in the generated list are in the template list
                    if not isinstance(template_value, list) or not all(item in template_value for item in value):
                        static_match = False
                        break
                elif template_value != value:
                    static_match = False
                    break
            
            # Check if surrounding context matches exactly
            surrounding_match = True
            for key, value in generated_surrounding.items():
                template_value = template['surrounding_context'].get(key)
                if isinstance(value, list):
                    # For list values, check if all elements in the generated list are in the template list
                    if not isinstance(template_value, list) or not all(item in template_value for item in value):
                        surrounding_match = False
                        break
                elif template_value != value:
                    surrounding_match = False
                    break
            
            # Only count if both static and surrounding match exactly
            if static_match and surrounding_match:
                match_count += 1
                
        return match_count
    
    def _save_destination_image(self, camera_orientation: List[float], camera_position: List[float], unrealcv_client: UnrealCV, save_path: str):
        """
        Save the destination image.
        """
        destination_image_name = "destination.png"
        landmark_path = os.path.join(save_path, destination_image_name)
        # time.sleep(0.2)
        unrealcv_client.set_location([camera_position[0], camera_position[1], 92.150003], "picture_man")
        unrealcv_client.custom_set_orientation([camera_orientation[0], camera_orientation[1], camera_orientation[2]], "picture_man")
        unrealcv_client.client.request("vset /camera/1/location {} {} {}".format(camera_position[0], camera_position[1], 200))
        unrealcv_client.client.request("vset /camera/1/rotation {} {} {}".format(0, camera_orientation[2], 0))

        time.sleep(0.5)
        # unrealcv_client.read_image(1, "lit", "file_path", landmark_path)
        img = unrealcv_client.read_image(1, "lit", "direct")
        cv2.imwrite(landmark_path, img)
        return img, [destination_image_name]

    def generate_natural_description(self, static_description: Dict, surrounding_context: Dict, img: np.ndarray) -> str:
        """Generate a natural language description from static description and surrounding context.
        
        Args:
            static_description: Dictionary containing static features of the asset
            surrounding_context: Dictionary containing surrounding environment information
            asset_type: Type of the asset (e.g., 'building', 'tree')
            
        Returns:
            str: Natural language description of the asset
        """
        def _process_image_to_base64(image: np.ndarray) -> str:
            import cv2
            import io
            from PIL import Image
            import base64
            import numpy as np
            """Convert numpy array image to base64 string.

            Args:
                image (np.ndarray): Image array (1 or 3 channels)

            Returns:
                str: Base64 encoded image string
            """
            # Convert single channel to 3 channels if needed
            if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Ensure uint8 type
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            # Convert to PIL Image
            pil_image = Image.fromarray(image)

            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_str}"
        
        model = CustomizedOpenAIModel("gpt-4o")

        response = model.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that writes short, natural directions to help someone find an urban building. "
                        "You will receive a front‑facing street‑level photo of the building and a structured summary of its features and nearby context.\n\n"
                        "Your job is to produce 2–3 friendly sentences that guide a person (or low‑height robot) to spot this building in real life. "
                        "Mention 1–2 of the building’s most distinctive features at eye‑level and reference one clear, nearby visual clue—like a sign, bench, lamp post, or neighboring building. "
                        "Avoid technical terms, measurements, or high‑up details like rooftops—focus on what’s visible from about 70 cm off the ground."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Here is a street‑facing image of the building." },
                        { "type": "image_url", "image_url": { "url": _process_image_to_base64(np.array(img)) } },
                        { "type": "text", "text": (
                            "Structured building info:\n" + json.dumps(static_description, indent=2)
                            + "\n\nNearby context:\n" + json.dumps(surrounding_context, indent=2)
                        ) }
                    ]
                }
            ]
        )
        return response
       

    def select_random_asset(self) -> str:
        """Select a random asset weighted by inverse frequency of asset types.
        
        The selection is weighted to favor rare asset types:
        1. Count frequency of each asset type
        2. Calculate inverse probability weights
        3. Choose random asset weighted by inverse frequency
        
        Returns:
            str: ID of the selected asset
        """
        # Count frequency of each asset type
        asset_counts = {}
        for asset_id in self.assets_template.keys():
            asset_type = self.assets_template[asset_id]['asset_type']
            asset_counts[asset_id] = sum(1 for a in self.assets_template.values() if a['asset_type'] == asset_type)
            
        # Calculate inverse probabilities based on frequency
        weights = [1.0/count for count in asset_counts.values()]
        weight_sum = sum(weights)
        normalized_weights = [w/weight_sum for w in weights]
        
        # Choose random asset weighted by inverse frequency
        asset_id = random.choices(list(self.assets_template.keys()), weights=normalized_weights, k=1)[0]
        return asset_id

    def generate_template_description(self, asset_id: str):
        """Generate a description template for a randomly selected asset.
        
        Uses select_random_asset() to choose an asset, then generates description by:
        1. Getting the full static description
        2. Getting the full surrounding context
        3. Counting exact matches in original data
        
        Returns:
            tuple: Contains:
                - static_description (dict): Static description elements
                - surrounding_context (dict): Surrounding context elements
                - exact_matches (int): Number of exact matches in original data
                - asset_id (str): ID of selected asset
        """

        template = self.assets_template[asset_id]
        
        # Get static description elements
        filtered_static = template['static_description']
        
        # Get surrounding context elements  
        filtered_surrounding = template['surrounding_context']
        
        # Count exact matches in the original data
        exact_matches = self._count_exact_template_matches(filtered_static, filtered_surrounding)
        
        return filtered_static, filtered_surrounding, exact_matches
    
    def _get_building_node(self, nodes: List[Node], front_door_position: List[float]) -> Node:
        """
        Get the building node for a given asset ID.
        """
        distances = [(((n.position.x - front_door_position[0])**2 + (n.position.y - front_door_position[1])**2)**0.5, n) for n in nodes]
        closest_nodes = sorted(distances, key=lambda x: x[0])[:3]

        for _, node in closest_nodes:
            if node.type == "normal":
                building_node = node
                break
        return building_node

    def generate_agent_spawning_transformations(self, building_node: Node, distance_units_to_building: List[int]) -> List[Point]:
        def _get_closest_nodes(map: Map, building_node: Node, distance_units_to_building: int = 20) -> List[Node]:
            """
            Get the closest nodes to the building node that are at least distance_units_to_building units away.
            """
            # normal_nodes = []
            # end_nodes = map.get_random_node_with_edge_distance([building_node], min_distance=distance_units_to_building, max_distance=distance_units_to_building)
            # for node in end_nodes:
            #     if node.type=="normal":
            #         normal_nodes.append((distance_units_to_building, node))
            end_nodes = map.get_random_node_with_edge_distance([building_node], min_distance=distance_units_to_building, max_distance=distance_units_to_building)
            return end_nodes
        """
        Generate agent spawning locations for a given asset.
        """
        spawning_locations = []
        spawning_orientations = []
        orientation_range = [0, 90, 180, 270]

        for distance in distance_units_to_building:
            normal_nodes = _get_closest_nodes(self.map, building_node, distance)
            if len(normal_nodes) >= 1:
                random_node = random.choice(normal_nodes)
                spawning_locations.append(random_node.position)
                spawning_orientations.append(random.choice(orientation_range))
            else:
                raise ValueError(f"No nodes found for distance {distance}")
            
        return spawning_locations, spawning_orientations
    
    def _generate_task_config(self, spawning_locations: List[Point], spawning_orientations: List[float], destination: Point, instruction: str, max_steps: int, dist_threshold: int, name_of_landmark_images: str):
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
                "destination": {
                    "x": destination[0],
                    "y": destination[1],
                    "z": 73.957685
                },
                "spawning_point": {
                    "location": {
                        "x": spawning_loc.x,
                        "y": spawning_loc.y,
                        "z": 73.957685
                    },
                    "orientation": {
                        "roll": 0, "pitch": 0, "yaw": spawning_orient
                    }
                },
                "instruction": instruction,
                "landmark_images": name_of_landmark_images
            }

        return {
            "criteria": {
                "max_steps": max_steps,
                "dist_threshold": dist_threshold
            },
            "agents": [
                _make_agent_config(i, loc, orient) 
                for i, (loc, orient) in enumerate(zip(spawning_locations, spawning_orientations))
            ]
        }
    
    
    def generate_task(self, output_path: str, distance_units_to_building: List[int]) -> str:
        """
        Generate a task for the agent.
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Select random asset
        asset_id = self.select_random_asset()
        static_description, surrounding_context, exact_matches = self.generate_template_description(asset_id)
        building_node = self._get_building_node(self.map.nodes, self.assets[asset_id]['front_door_position'])
        print("building_node:", building_node)
        spawning_locations, spawning_orientations = self.generate_agent_spawning_transformations(building_node, distance_units_to_building)
        print("spawning_locations:", spawning_locations)
        print("spawning_orientations:", spawning_orientations)


        img, image_path_name = self._save_destination_image(self.assets[asset_id]['camera_orientation'], self.assets[asset_id]['camera_position'], self.unrealcv_client, output_path)
        print("image_path_name:", image_path_name)
        description = self.generate_natural_description(static_description, surrounding_context, img)
        print("description:", description)
        task_config = self._generate_task_config(spawning_locations, spawning_orientations, self.assets[asset_id]['front_door_position'], description, 150, 200, image_path_name)
        print("task_config:", task_config)
        with open(os.path.join(output_path, "task_config.json"), "w") as f:
            json.dump(task_config, f)
        print("task_config saved")
        output_img_path = os.path.join(output_path, "map.png")
        destination_position = Vector(self.assets[asset_id]['front_door_position'][0], self.assets[asset_id]['front_door_position'][1])
        visualize_map_with_multiagents(self.map, spawning_locations, destination_position, save_path=output_img_path)

        return img, static_description, surrounding_context, description
    
    def exit(self):
        self.unrealcv_client.client.disconnect()
    