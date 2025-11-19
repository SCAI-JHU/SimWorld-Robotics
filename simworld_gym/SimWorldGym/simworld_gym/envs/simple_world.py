import gym
import os
import time
from datetime import datetime
from gym import spaces
import numpy as np
from simworld_gym.utils.unrealcv_basic import UnrealCV
from simworld_gym.utils import misc
from simworld_gym.utils.environment_generator import EnvironmentGenerator
from simworld_gym.utils.agent_controller import AgentController
import cv2
import csv
import re
import json
import math
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.ops import unary_union

try:
    from IPython import get_ipython
    from IPython.display import display, Image
    JUPYTER = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
except:
    JUPYTER = False


class SimpleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 30}
    def __init__(self, action_config="action_config.json", render_mode=None, observation_type="ground_truth", record_video=False, log_dir=None, reward_setting=None, resolution=(320, 240), port=9000):
        # Connect to UnrealCV server
        port = port
        env_ip = "127.0.0.1"
        self.resolution = resolution
        self.unrealcv = UnrealCV(port=port, ip=env_ip, resolution=resolution)
        
        # Initialize environment components
        self.environment_generator = EnvironmentGenerator(self.unrealcv)
        self.agent_controller = AgentController(self.unrealcv, resolution)

        # Validate and set render mode
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # Initialize environment state
        self.launched = False
        self.steps_count = 0
        self.substeps_count = 0
        self.subtask_max_steps = 0

        # Set observation type
        self.observation_type = observation_type
        assert self.observation_type in ['ground_truth', 'rgb', 'depth', 'rgbd', 'all'], \
            f"Invalid observation type: {observation_type}. Must be one of: ground_truth, rgb, depth, rgbd, all"

        # Load action space configuration
        if action_config:
            self.action_config = misc.load_config(action_config)
            self.agent_controller.action_config = self.action_config
            # Define action space based on configuration
            if self.action_config["action_space"]["type"] == "discrete":
                self.action_space = spaces.Discrete(self.action_config["action_space"]["n"])
                self.action_meanings = self.action_config["action_meanings"]
            elif self.action_config["action_space"]["type"] == "continuous":
                # Add support for continuous action space if needed
                raise NotImplementedError("Continuous action space not yet supported")
        else:
            self.action_space = spaces.Discrete(6)  # Updated to match action_config.json

        # Define observation space based on observation type
        if self.observation_type == 'ground_truth':
            self.observation_space = spaces.Dict({
                "relative_target_location": spaces.Box(-1000, 1000, shape=(3,), dtype=np.float64),
                "relative_target_orientation": spaces.Box(-1000, 1000, shape=(1,), dtype=np.float64),
            })
        else:
            # Image-based observation spaces
            if self.observation_type == 'rgb':
                self.observation_space = spaces.Dict({
                    "rgb": spaces.Box(0, 255, shape=(resolution[1], resolution[0], 3), dtype=np.uint8)
                })
            elif self.observation_type == 'depth':
                self.observation_space = spaces.Dict({
                    "depth": spaces.Box(0, 255, shape=(resolution[1], resolution[0], 1), dtype=np.uint8)
                })
            elif self.observation_type == 'rgbd':
                self.observation_space = spaces.Dict({
                    "rgb": spaces.Box(0, 255, shape=(resolution[1], resolution[0], 3), dtype=np.uint8),
                    "depth": spaces.Box(0, 255, shape=(resolution[1], resolution[0], 1), dtype=np.uint8)
                })
            elif self.observation_type == 'all':
                self.observation_space = spaces.Dict({
                    "rgb": spaces.Box(0, 255, shape=(resolution[1], resolution[0], 3), dtype=np.uint8),
                    "depth": spaces.Box(0, 255, shape=(resolution[1], resolution[0], 1), dtype=np.uint8),
                    "object_mask": spaces.Box(0, 255, shape=(resolution[1], resolution[0], 3), dtype=np.uint8)
                })

        # Initialize environment configuration
        self.agent_json = None
        self.world_json = None

        # Initialize agent and target state
        self._target_location = None
        self._agent_location = None
        self._agent_rotation = None
        self._criteria = None

        self._previous_agent_location = None
        self._moving_distance = 0

        # Setup logging directory with default path
        if log_dir is None:
            self.log_dir = os.path.join(os.getcwd(), "logs")
        else:
            self.log_dir = os.path.join(log_dir, "logs")
        
        self.env_name = self.__class__.__name__.lower()
        self.has_traffic = False
        
        self.reset_count = 0
        self.start_time = None
        self.trajectory_writer = None
        self.images_dir = None

        # Record video settings
        self.record_video = record_video
        self.record_video_fps = None
        if record_video:
            self.record_video_fps = 8
            self.unrealcv.set_fps(self.record_video_fps)
        else:
            # If not record video, set fps to 100
            self.unrealcv.set_fps(100)
        # Create base directory
        self.base_dir = misc.create_experiment_dir(self.log_dir, self.env_name)

        self.total_human_collision = 0
        self.total_object_collision = 0
        self.total_building_collision = 0
        self.total_vehicle_collision = 0
        self.current_human_collision = 0
        self.current_object_collision = 0
        self.current_building_collision = 0
        self.current_vehicle_collision = 0
        self.human_collision_penalty = reward_setting.get("human_collision_penalty", -0.5)
        self.object_collision_penalty = reward_setting.get("object_collision_penalty", -0.1)
        self.building_collision_penalty = -0.1
        self.vehicle_collision_penalty = -0.5

        self.action_penalty = reward_setting.get("action_penalty", -0.01)
        self.success_reward = reward_setting.get("success_reward", 10)
        self.off_track_penalty = reward_setting.get("off_track_penalty", -0.1)

        #store last 10 positions of the agent
        self.last_twenty_positions = []

    def _get_obs(self):
        if self.observation_type == 'ground_truth':
            relative_target_location = self._target_location - self._agent_location
            relative_target_orientation = (np.degrees(np.arctan2(relative_target_location[1], relative_target_location[0]))-self._agent_rotation[1])%360
            if relative_target_orientation > 180:
                relative_target_orientation -= 360
            return {
                "relative_target_location": relative_target_location,
                "relative_target_orientation": relative_target_orientation
            }
        else:
            # Image-based observations
            if self.observation_type == 'rgb':
                return {
                    "rgb": self.agent_controller.get_image("lit", "direct")
                }
            elif self.observation_type == 'depth':
                return {
                    "depth": self.agent_controller.get_image("depth", "direct")
                }
            elif self.observation_type == 'rgbd':
                return {
                    "rgb": self.agent_controller.get_image("lit", "direct"),
                    "depth": self.agent_controller.get_image("depth", "direct")
                }
            elif self.observation_type == 'all':
                return {
                    "rgb": self.agent_controller.get_image("lit", "direct"),
                    "depth": self.agent_controller.get_image("depth", "direct"),
                    "object_mask": self.agent_controller.get_image("object_mask", "direct"),
                }

    def _get_infor(self):
        self.total_human_collision += self.current_human_collision
        self.total_object_collision += self.current_object_collision
        self.total_building_collision += self.current_building_collision
        self.total_vehicle_collision += self.current_vehicle_collision
        angle = self._agent_rotation[1] % 360
        if angle < 0:
            angle += 360
            
        if 175 <= angle <= 185:
            rotation = "East"
        elif angle >= 355 or angle <= 5:
            rotation = "West"
        elif 85 <= angle <= 95:
            rotation = "North"
        elif 265 <= angle <= 275:
            rotation = "South"
        else:
            angles = [180, 0, 90, 270]  # West, East, North, South
            directions = ["East", "West", "North", "South"]
            closest_idx = min(range(len(angles)), key=lambda i: abs(angle - angles[i]))
            rotation = directions[closest_idx]
        return {
            "total_step": self.steps_count,
            "success": self._get_success(),
            "agent": {
                "agent_location": self._agent_location,
                "agent_rotation": rotation
            },
            "target": {
                "target_location": self._target_location
            },
            "agent_collision": {
                "human": self.total_human_collision,
                "object": self.total_object_collision,
                "building": self.total_building_collision,
                "vehicle": self.total_vehicle_collision
            },
            "moving_distance": self._moving_distance,
            "current_instruction": self.current_instruction
        }
    
    def _get_reward(self):
        if self._get_success():
            return self.success_reward
        else:
            collision_penalty = self.current_human_collision * self.human_collision_penalty + self.current_object_collision * self.object_collision_penalty
            action_penalty = self.action_penalty
            off_track_penalty = self.off_track_penalty * (
                np.linalg.norm(self._agent_location[:2] - self._target_location[:2]) - 
                np.linalg.norm(self._previous_agent_location[:2] - self._target_location[:2])
                )

            return collision_penalty + action_penalty + off_track_penalty

    def _get_terminated(self):
        if (self._get_success()) or (self.steps_count >= self._criteria["max_steps"]):
            return True
        else:
            return False

    def _get_success(self):
        if self.current_subtask < len(self.total_instruction):
            return False
        else:
            return True

    @staticmethod
    def build_road_polygon(world_json_path, road_length=22000, road_width=42000):
        with open(world_json_path, 'r') as f:
            data = json.load(f)

        polygons = []

        for node in data['nodes']:
            node_id = node.get('id', '')
            if node_id.startswith('GEN_Road_'):
                try:
                    road_index = int(node_id.split('_')[-1])
                except ValueError:
                    continue
                if 0 <= road_index < 20:
                    props = node['properties']
                    loc = props['location']
                    yaw_deg = props['orientation']['yaw']
                    yaw_rad = math.radians(yaw_deg)

                    # Road center
                    cx, cy = loc['x'], loc['y']

                    # Half-dimensions
                    half_length = road_length / 2
                    half_width = road_width / 2

                    # Compute corners around center based on yaw
                    dx = math.cos(yaw_rad)
                    dy = math.sin(yaw_rad)

                    perp_dx = -dy
                    perp_dy = dx

                    corners = []
                    for sx, sy in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
                        px = cx + sx * half_length * dx + sy * half_width * perp_dx
                        py = cy + sx * half_length * dy + sy * half_width * perp_dy
                        corners.append((px, py))

                    poly = Polygon(corners)
                    polygons.append(poly)

        return unary_union(polygons)
    
    @staticmethod
    def is_robot_inside_complex(x, y, world_polygon):
        point = Point(x, y)
        return world_polygon.contains(point)

    def _get_terminated(self):
        # Check if the agent is out of the world bounding box
        if not self.is_robot_inside_complex(self._agent_location[0], self._agent_location[1], self.world_bounding_box):
            return True
        
        # Check if the agent has repeated the last 20 positions
        if len(self.last_twenty_positions) >= 20:
            # Convert positions to numpy array for vectorized operations
            positions = np.array(self.last_twenty_positions)
            first_pos = positions[0]
            
            # Calculate distances using vectorized operations
            distances = np.sqrt(np.sum((positions - first_pos)**2, axis=1))
            
            # Check if all distances are within threshold
            threshold = 1  # 10cm threshold
            if np.all(distances <= threshold):
                return True

        if (self._get_success()) or (self.steps_count >= self._criteria["max_steps"]) or (self.substeps_count >= self.subtask_max_steps):
            return True
        return False

    def _get_evaluation_of_subtask(self):
        subtask_evaluation_criteria = self.total_evaluation_of_subtask[self.current_subtask]
        print("\n=== Subtask Evaluation Information ===")
        print(f"Current subtask: {self.current_subtask}")
        print(f"Evaluation criteria type: {subtask_evaluation_criteria.get('type')}")
        print(f"Current agent location: {self._agent_location[:2]}")
        print(f"Current agent rotation: {self._agent_rotation[1]}")
        print(f"Distance threshold: {self._criteria['dist_threshold']}")
        
        if subtask_evaluation_criteria.get("type") == "ORIENTATION":
            print("\n=== ORIENTATION Evaluation ===")
            print(f"Target orientation: {subtask_evaluation_criteria.get('correct_robot_orientation')}")
            print(f"Current orientation: {self._agent_rotation[1]}")
            print(f"Orientation difference: {(subtask_evaluation_criteria.get('correct_robot_orientation') - self._agent_rotation[1]) % 360}")
            if (subtask_evaluation_criteria.get("correct_robot_orientation") - self._agent_rotation[1]) % 360 < 45:
                print("Orientation check passed")
                return True
            else:
                print("Orientation check failed")
                return False
        elif subtask_evaluation_criteria.get("type") == "ROAD":
            print("\n=== ROAD Evaluation ===")
            print(f"Target location: {subtask_evaluation_criteria.get('correct_robot_location')}")
            print(f"Current location: {self._agent_location[:2]}")
            print(f"Distance: {np.linalg.norm(np.array(subtask_evaluation_criteria.get('correct_robot_location')) - self._agent_location[:2], ord=1)}")
            if np.linalg.norm(np.array(subtask_evaluation_criteria.get("correct_robot_location")) - self._agent_location[:2], ord=1) < self._criteria["dist_threshold"]:
                print("Distance check passed")
                return True
            else:
                print("Distance check failed")
                return False
        elif subtask_evaluation_criteria.get("type") == "TURNING":
            print("\n=== TURNING Evaluation ===")
            print(f"Target orientation: {subtask_evaluation_criteria.get('correct_robot_orientation')}")
            print(f"Current orientation: {self._agent_rotation[1]}")
            print(f"Orientation difference: {(subtask_evaluation_criteria.get('correct_robot_orientation') - self._agent_rotation[1]) % 360}")
            if (subtask_evaluation_criteria.get("correct_robot_orientation") - self._agent_rotation[1]) % 360 < 45:
                print("Orientation check passed")
                if subtask_evaluation_criteria.get("correct_robot_position_area", None) is None:
                    print("No location check required")
                    return True
                else:
                    print(f"Target area: {subtask_evaluation_criteria.get('correct_robot_position_area')}")
                    print(f"Current location: {self._agent_location[:2]}")
                    min_x, min_y = subtask_evaluation_criteria["correct_robot_position_area"][0]
                    max_x, max_y = subtask_evaluation_criteria["correct_robot_position_area"][1]
                    x, y = self._agent_location[:2]
                    if min_x <= x <= max_x and min_y <= y <= max_y:
                        print("Position check passed")
                        return True
                    else:
                        print("Position check failed")
                        return False
            else:
                print("Orientation check failed")
                return False
        elif subtask_evaluation_criteria.get("type") == "FINISH":
            print("\n=== FINISH Evaluation ===")
            print(f"Target location: {subtask_evaluation_criteria.get('correct_robot_location')}")
            print(f"Current location: {self._agent_location[:2]}")
            print(f"Distance: {np.linalg.norm(np.array(subtask_evaluation_criteria.get('correct_robot_location')) - self._agent_location[:2], ord=1)}")
            if np.linalg.norm(np.array(subtask_evaluation_criteria.get("correct_robot_location")) - self._agent_location[:2], ord=1) < self._criteria["dist_threshold"]:
                print("Distance check passed")
                print(f"Target orientation: {subtask_evaluation_criteria.get('correct_robot_orientation')}")
                print(f"Current orientation: {self._agent_rotation[1]}")
                print(f"Orientation difference: {(subtask_evaluation_criteria.get('correct_robot_orientation') - self._agent_rotation[1]) % 360}")
                if (subtask_evaluation_criteria.get("correct_robot_orientation") - self._agent_rotation[1]) % 360 < 45:
                    print("Orientation check passed")
                    return True
                else:
                    print("Orientation check failed") 
                    return False
            else:
                print("Distance check failed")
                return False
    def _get_instructions(self):
        return self._criteria["instruction"]
    
    def reset(self, options=None, seed=None):
        super().reset(seed=seed, options=options)

        if options is not None:
            if "task_path" in options:
                self.task_path = options["task_path"]
            world, task = os.path.split(self.task_path)
            _, world = os.path.split(world)
            if "world_json" in options:
                self.world_json = options["world_json"]
                self.environment_generator.loading_json(self.world_json)
                self.environment_generator.clear_env()
                self.environment_generator.generate_world()
                print("Reset world")
            
            if "agent_json" in options:
                self.agent_json =  misc.load_env_setting(options["agent_json"])
                self.agent_controller.reset(self.agent_json)
                print("Reset agent")
            else:
                self.agent_controller.reset()
            
            # if "traffic_json" in options:
            #     if not self.has_traffic:
            #         traffic_json = misc.load_config(options["traffic_json"])
            #         num_vehicles = traffic_json["num_vehicles"]
            #         num_pedestrians = traffic_json["num_pedestrians"]
            #         map = traffic_json["map"]
            #         seed = traffic_json["seed"]
            #         dt = traffic_json["dt"]
            #         traffic_generator = TrafficGenerator(num_vehicles, num_pedestrians, map, seed, dt)
            #         self.traffic_generator = traffic_generator
            #     else:
            #         self.traffic_generator.reset()
            #     self.has_traffic = True
            #     print("Reset traffic")
            
        else:
            self.agent_controller.reset()

        # Close previous trajectory file if open
        if self.trajectory_writer is not None:
            self.trajectory_writer.close()
        
        # Update reset count and create new directory structure
        self.reset_count += 1

        self._moving_distance = 0
        
        # Create new episode directory
        self.current_episode_dir = os.path.join(
            self.base_dir,
            f"reset_{self.reset_count}_{world}_{task}"
        )
        os.makedirs(self.current_episode_dir, exist_ok=True)
        
        # Create images directory for this episode
        self.images_dir = os.path.join(self.current_episode_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.start_time = time.perf_counter()
        
        # Initialize the agent and target state
        self.agent_controller.update_agent_transformation_hard()
        self._agent_location = self.agent_controller.return_agent_location()
        self._agent_rotation = self.agent_controller.return_agent_rotation()
        self._target_location = self.agent_controller.return_target_location()
        self._criteria = self.agent_controller.return_criteria()

        # Setup new trajectory file
        self.trajectory_file = os.path.join(self.current_episode_dir, "trajectory.csv")
        self.trajectory_writer = open(self.trajectory_file, 'w', newline='')
        self.csv_writer = csv.writer(self.trajectory_writer)
        
        # Write header
        header = ["step", "timestamp", "action", "action_meaning", "agent_x", "agent_y", "agent_z", "agent_rotation", 
                 "target_x", "target_y", "moving_distance", "human_collision", "object_collision", "building_collision", "vehicle_collision", "subtask_success", "success"]
        self.csv_writer.writerow(header)
        
        # Log the initial position and rotation of the agent
        initial_state = ["Initial Position", "", "", "", f"{self._agent_location[0]:.2f}", f"{self._agent_location[1]:.2f}", 
                        f"{self._agent_location[2]:.2f}", f"{self._agent_rotation[1]:.2f}", "", "", "0", "0", "0", "0", "0", "0"]
        self.csv_writer.writerow(initial_state)
        self.trajectory_writer.flush()

        self._previous_agent_location = self._agent_location

        self.steps_count = 0
        self.substeps_count = 0
        self.subtask_max_steps = 10

        if self.render_mode == "human":
            self._render_frame()


        self.total_evaluation_of_subtask = self.agent_controller.return_evaluation_of_subtask()
        self.total_instruction = self.agent_controller.return_instruction()
        self.total_instruction_images = []

        agent_instruction_images_path = self.agent_controller.return_instruction_image_path()

        for image_path in agent_instruction_images_path:
            path = misc.get_settingpath(os.path.join(self.task_path, image_path))
            self.total_instruction_images.append(cv2.imread(path))

        self.current_subtask = 0
        self.current_instruction = {}
        self.current_instruction["text"] = self.total_instruction[self.current_subtask]
        self.current_instruction["image"] = self.total_instruction_images[self.current_subtask]

        observation = self._get_obs()
        info = self._get_infor()
        
        self.world_bounding_box = self.build_road_polygon(misc.get_settingpath(self.world_json))
        #adding the current position to the last ten positions
        #store last 10 positions of the agent
        self.last_twenty_positions = []
        self.last_twenty_positions.append(self._agent_location[:2])

        self.total_human_collision = 0
        self.total_object_collision = 0
        self.total_building_collision = 0
        self.total_vehicle_collision = 0

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            observation: The current observation
            reward: The reward received
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Read Action from the buffer
        # Use the agent controller to perfrom the action
        action_start_time = time.perf_counter() - self.start_time
        if action == -1:
            self._evaluation_of_subtask = self._get_evaluation_of_subtask()
            self.current_instruction = {}
            if self._evaluation_of_subtask:
                self.current_subtask += 1
                if self.current_subtask < len(self.total_instruction):
                    self.current_instruction["text"] = self.total_instruction[self.current_subtask]
                    self.current_instruction["image"] = self.total_instruction_images[self.current_subtask]
                    subtask_evaluation_criteria = self.total_evaluation_of_subtask[self.current_subtask]
                    if subtask_evaluation_criteria.get("type") == "ORIENTATION":
                        self.subtask_max_steps = 10
                    elif subtask_evaluation_criteria.get("type") == "TURNING":
                        self.subtask_max_steps = 50
                    else:
                        current_location = self._agent_location[:2]
                        target_location = subtask_evaluation_criteria.get("correct_robot_location")
                        expected_distance = np.linalg.norm(np.array(target_location) - np.array(current_location), ord=1)
                        self.subtask_max_steps = int(expected_distance / 100)
                    self.substeps_count = 0
                    terminated = False
                else:
                    terminated = True
            else:
                terminated = True
        else:
            performing_time = self.agent_controller.apply_action(action)            
            if self.record_video:
                action_dir = os.path.join(self.images_dir, f"action_{self.steps_count:06d}")
                self.agent_controller.render_video(performing_time, self.record_video_fps, "lit", action_dir)
            else:
                time.sleep(performing_time+0.05)
                # Wait until the action in Unreal Engine has finished executing
                self.agent_controller.update_agent_availability_hard()
                while not self.agent_controller.return_agent_availability():
                    self.agent_controller.update_agent_availability_hard()
                    time.sleep(0.05)
            # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            # Save the observation image after executing the action
            if self.record_video:
                observation_image_path = os.path.join(self.images_dir, f"observation_{self.steps_count:06d}.png")
                self.agent_controller.get_image("lit", "file_path", observation_image_path)
            
            self.agent_controller.update_agent_transformation_hard()
            self._agent_location = self.agent_controller.return_agent_location()
            self._agent_rotation = self.agent_controller.return_agent_rotation()

            if self._previous_agent_location is not None:
                self._moving_distance += np.linalg.norm(self._agent_location[:2] - self._previous_agent_location[:2])
            #add the current position to the last ten positions after checking the length of the list
            if len(self.last_twenty_positions) >= 20:
                self.last_twenty_positions.pop(0)
            self.last_twenty_positions.append(self._agent_location[:2])
            terminated = self._get_terminated()
        
        self.steps_count += 1
        self.substeps_count += 1
        observation = self._get_obs()

        self.current_human_collision, self.current_object_collision, self.current_building_collision, self.current_vehicle_collision = self.agent_controller.get_agent_collision()
        reward = self._get_reward()
        info = self._get_infor()

        if self.trajectory_writer is not None:
            self._write_trajectory(self.trajectory_writer, 
                                self.steps_count, 
                                action_start_time, 
                                action, 
                                self._agent_location, 
                                self._agent_rotation, 
                                self._target_location, 
                                self._moving_distance, 
                                self.current_human_collision, 
                                self.current_object_collision, 
                                self.current_building_collision,
                                self.current_vehicle_collision,
                                self.current_subtask,
                                self._get_success())
        
        self._previous_agent_location = self._agent_location

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        img = self.agent_controller.get_image("lit", "direct")
        if self.render_mode == "human":
            if JUPYTER:
                # Convert to RGB for display
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Create a temporary file for display
                temp_path = os.path.join(self.log_dir, "temp_display.png")
                cv2.imwrite(temp_path, img_rgb)
                display(Image(filename=temp_path))
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                cv2.imshow("CityNav Environment", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
        return img
    
    def _write_trajectory(self, trajectory_writer, steps, action_start_time, action, agent_location, agent_rotation, target_location, moving_distance, human_collision, object_collision, building_collision, vehicle_collision, current_subtask, success):
        if trajectory_writer is not None:
            action_meaning = self.action_meanings[str(action)] if action != -1 else "EVALUATE"
            # 使用csv.writer写入数据
            csv_writer = csv.writer(trajectory_writer)
            csv_writer.writerow([
                steps,
                action_start_time,
                action,
                action_meaning,
                f"{agent_location[0]:.2f}",
                f"{agent_location[1]:.2f}",
                f"{agent_location[2]:.2f}",
                f"{agent_rotation[1]:.2f}",
                f"{target_location[0]:.2f}",
                f"{target_location[1]:.2f}",
                moving_distance,
                human_collision,
                object_collision,
                building_collision,
                vehicle_collision,
                current_subtask,
                success
            ])
            trajectory_writer.flush()

    def close(self):
        if self.trajectory_writer is not None:
            self.trajectory_writer.flush()
            self.trajectory_writer.close()
            self.trajectory_writer = None
        
        cv2.destroyAllWindows()  # Close any open windows
        self.unrealcv.client.disconnect()
