import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
import re
import time
import csv
import cv2
import json

from simworld_gym.utils.unrealcv_basic import UnrealCV
from simworld_gym.utils import misc
from simworld_gym.utils.environment_generator import EnvironmentGenerator
from simworld_gym.utils.multi_agent_controller import MultiAgentController
from simworld_gym.utils.base import ActionBuffer



class BufferEnv(gym.Env):
    """
    A gym environment for multi-agent navigation tasks. Each agent operates independently but shares the same environment.
    The environment uses a buffer system to handle multiple agent actions efficiently.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 30}
    def __init__(self, buffer_max_size=None, action_config=None, render_mode=None, observation_type="ground_truth", record_video=False, log_dir=None, reward_setting={}, resolution=(320, 240), debug=False, port=9000, env_ip="127.0.0.1"):
        # Connect to UnrealCV server
        env_ip = "127.0.0.1"
        self.resolution = resolution
        self.unrealcv = UnrealCV(port=port, ip=env_ip, resolution=resolution)
        
        # Initialize environment components
        self.environment_generator = EnvironmentGenerator(self.unrealcv)
        self.multi_agent_controller = MultiAgentController(self.unrealcv, self.resolution)   

        # Validate and set render mode
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        # Initialize environment state
        self.launched = False
        self.steps_count = 0

        # Set observation type
        self.observation_type = observation_type
        assert self.observation_type in ['ground_truth', 'rgb', 'depth', 'rgbd', 'all'], \
            f"Invalid observation type: {observation_type}. Must be one of: ground_truth, rgb, depth, rgbd, all"

        # Add logging setup
        if log_dir is None:
            self.log_dir = os.path.join(os.getcwd(), "logs")
        else:
            self.log_dir = os.path.join(log_dir, "logs")
        
        self.env_name = self.__class__.__name__.lower()
        self.reset_count = 0
        self.start_time = None
        self.trajectory_writers = {}
        self.images_dir = None

        # Record video settings
        self.record_video = record_video
        self.record_video_fps = None
        if record_video:
            self.record_video_fps = 8
            self.unrealcv.set_fps(self.record_video_fps)
        else:
            self.unrealcv.set_fps(100)

        # Create base directory
        self.base_dir = misc.create_experiment_dir(self.log_dir, self.env_name)

        # Load action space configuration
        if action_config:
            self.action_config = misc.load_config(action_config)
            self.multi_agent_controller.action_config = self.action_config
            # Define action space based on configuration
            if self.action_config["action_space"]["type"] == "discrete":
                self.action_space = spaces.Discrete(self.action_config["action_space"]["n"])
                self.action_meanings = self.action_config["action_meanings"]
            elif self.action_config["action_space"]["type"] == "continuous":
                raise NotImplementedError("Continuous action space not yet supported")
        else:
            self.action_space = spaces.Discrete(6) 

        # Define observation space
        if self.observation_type == 'ground_truth':
            self.observation_space = spaces.Dict({
                "agent": spaces.Box(-1000, 1000, shape=(2,), dtype=np.float64),
                "target": spaces.Box(-1000, 1000, shape=(2,), dtype=np.float64),
            })
        else:
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
        self.action_buffer = ActionBuffer(max_size=buffer_max_size, unrealcv_client=self.unrealcv)
        self.num_agents = 0

        self.debug = debug

        self.start_time = None
        self.map_path = None

        # Initialize movement tracking
        self._previous_agent_location = None

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

    def _get_obs(self, agent_indexes):
        if self.observation_type == 'ground_truth':
            return {
                "agents": self._agent_location,
                "targets": self._target_location
            }
        else:
            if self.observation_type == 'rgb':
                return {"rgb": self.multi_agent_controller.get_image(agent_indexes, "lit", "direct")}
            elif self.observation_type == 'depth':
                return {"depth": self.multi_agent_controller.get_image(agent_indexes, "depth", "direct")}
            elif self.observation_type == 'object_mask':
                return {"object_mask": self.multi_agent_controller.get_image(agent_indexes, "object_mask", "direct")}
            elif self.observation_type == 'rgbd':
                return {
                    "rgb": self.multi_agent_controller.get_image(agent_indexes, "lit", "direct"),
                    "depth": self.multi_agent_controller.get_image(agent_indexes, "depth", "direct")
                }
            elif self.observation_type == 'all':
                return {
                    "rgb": self.multi_agent_controller.get_image(agent_indexes, "lit", "direct"),
                    "depth": self.multi_agent_controller.get_image(agent_indexes, "depth", "direct"),
                    "object_mask": self.multi_agent_controller.get_image(agent_indexes, "object_mask", "direct"),
                }

    def _transform_rotation(self, rotation):
        angle = rotation[1]
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
        return rotation
    
    def _contains_color(self, image_np, target_color):
        transfer_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        target_color = np.array(target_color).reshape(1, 1, 3)
        match = np.all(transfer_image == target_color, axis=-1)
        return np.any(match)
    
    def _check_has_robot_in_view(self, image):
        robot_color = [0, 255, 128]
        return self._contains_color(image, robot_color)
    
    def _get_infor(self):
        self.total_human_collision += self.current_human_collision
        self.total_object_collision += self.current_object_collision
        self.total_building_collision += self.current_building_collision
        self.total_vehicle_collision += self.current_vehicle_collision
        # print(f"self._agent_rotation: {self._agent_rotation}")

        return {
            "total_step": self.steps_count,
            "success": self._get_success(),
            "agent": {
                "agent_location": self._agent_location,
                "agent_rotation": [self._transform_rotation(rotation) for rotation in self._agent_rotation],
                "agent_orientation": self._agent_rotation
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
            "moving_distance": self._moving_distance
        }

    def _get_terminated(self):
        if (np.any(self.steps_count >= self._criteria["max_steps"])):
            return True
        else:
            return False

    def _get_success(self):
        return self.success
        

    def _get_reward(self):
        """为每个智能体计算独立奖励"""
        rewards = np.zeros(self.num_agents)
        
        # 计算每个智能体的成功状态
        success = self._get_success()
        
        # 成功奖励
        rewards[success] = self.success_reward
        
        # 未成功智能体的惩罚
        mask = ~success
        rewards[mask] = (
            self.action_penalty +
            self.human_collision_penalty * self.current_human_collision[mask] +
            self.object_collision_penalty * self.current_object_collision[mask]
        )
        
        return rewards

    def reset(self, options=None, seed=None):
        # We can pass the change of the gym in this option parameter to modify the initial condition, like the world setting or the agent setting.
        super().reset(seed=seed, options=options)
        if options is not None:
            if "task_path" in options:
                self.task_path = options["task_path"]

            if "map_path" in options:
                self.map_path = options["map_path"]
            
            if "world_json" in options:
                self.world_json = options["world_json"]
                self.environment_generator.loading_json(self.world_json)
                self.environment_generator.clear_env()
                self.environment_generator.generate_world()
                print("environment reset")
            
            if "agent_json" in options:
                self.agent_json = misc.load_env_setting(options["agent_json"])
                self.num_agents = len(self.agent_json['agents'])
                self.multi_agent_controller.reset(self.agent_json)
                print("agent reset")
            else:
                self.multi_agent_controller.reset()
                print("agent reset")


        else:
            self.multi_agent_controller.reset()

        # Initialize high-dimensional variables based on num_agents
        self._target_location = np.zeros((self.num_agents, 3))
        self._agent_location = np.zeros((self.num_agents, 3))
        self._agent_rotation = np.zeros((self.num_agents, 3))
        self._previous_agent_location = np.zeros((self.num_agents, 3))
        self._moving_distance = np.zeros(self.num_agents)
        self.total_human_collision = np.zeros(self.num_agents)
        self.total_object_collision = np.zeros(self.num_agents)

        # Close previous trajectory files if open
        if hasattr(self, 'trajectory_writers'):
            for writer in self.trajectory_writers.values():
                if writer is not None:
                    writer.close()
            self.trajectory_writers = {}
        
        # Update reset count and create new directory structure
        self.reset_count += 1
        
        # Create new episode directory
        self.current_episode_dir = os.path.join(
            self.base_dir,
            f"reset_{self.reset_count}"
        )
        os.makedirs(self.current_episode_dir, exist_ok=True)
        
        # Initialize new trajectory writers and images directories
        self.images_dirs = {}
        self.trajectory_writers = {}
        
        # Define header for trajectory files
        header = ["step", "timestamp", "action", "action_meaning", "agent_x", "agent_y", "agent_z", "agent_rotation", "target_x", "target_y", "moving_distance", 
                  "human_collision", "object_collision", "building_collision", "vehicle_collision", "messages", "success"]
        
        for i in range(self.num_agents):
            # Create main agent directory
            agent_dir = os.path.join(self.current_episode_dir, f"agent_{i}")
            os.makedirs(agent_dir, exist_ok=True)
            
            # Create images directory with observation and action subdirectories
            images_dir = os.path.join(agent_dir, "images")
            self.images_dirs[i] = {
                'base': images_dir,
                'observations': os.path.join(images_dir, "observations"),
                'actions': os.path.join(images_dir, "actions")
            }
            
            # Create all necessary directories
            os.makedirs(self.images_dirs[i]['observations'], exist_ok=True)
            os.makedirs(self.images_dirs[i]['actions'], exist_ok=True)
            
            # Create trajectory file
            trajectory_file = os.path.join(agent_dir, "trajectory.csv")
            self.trajectory_writers[i] = open(trajectory_file, 'w', newline='')
            csv_writer = csv.writer(self.trajectory_writers[i])

            csv_writer.writerow(header)
            self.trajectory_writers[i].flush()
        
        self.multi_agent_controller.update_agent_transformation_hard(np.arange(self.num_agents))
        self._agent_location = self.multi_agent_controller.return_agent_location()
        self._agent_rotation = self.multi_agent_controller.return_agent_rotation()
        self._target_location = self.multi_agent_controller.return_target_location()
        self._criteria = self.multi_agent_controller.return_criteria()

        self.start_time = time.perf_counter()

        # Initialize collision tracking
        self.total_human_collision = np.zeros(self.num_agents)
        self.total_object_collision = np.zeros(self.num_agents)
        self.total_building_collision = np.zeros(self.num_agents)
        self.total_vehicle_collision = np.zeros(self.num_agents)
        self.current_human_collision = np.zeros(self.num_agents)
        self.current_object_collision = np.zeros(self.num_agents)
        self.current_building_collision = np.zeros(self.num_agents)
        self.current_vehicle_collision = np.zeros(self.num_agents)
        self._previous_agent_location = self._agent_location.copy()
        self._agent_previous_availability = np.zeros(self.num_agents, dtype=bool)

        self.agent_current_actions = np.zeros(self.num_agents, dtype=int)
        self.agent_current_actions_start_time = np.zeros(self.num_agents)

        self.agent_current_messages = [[] for _ in range(self.num_agents)]

        observation = self._get_obs(np.arange(self.num_agents))
        self.agent_last_observation = observation
        self.steps_count = np.zeros(self.num_agents)
        self.success = False
        info = self._get_infor()
        self.steps_count = np.zeros(self.num_agents)

        map_folder = misc.get_settingpath(os.path.join(self.map_path, self.agent_json["map_index"]))
        
        buildings_info = json.load(open(os.path.join(map_folder, "total_buildings.json")))
        landmark_folder = os.path.join(map_folder, "robot_view_images")
        landmark_files = os.listdir(landmark_folder)
        landmark_files = [file for file in landmark_files if file.endswith(".png")]
        landmark_files = [os.path.join(landmark_folder, file) for file in landmark_files]

        landmark_images = [cv2.imread(file) for file in landmark_files]

        if self.render_mode == "human":
            self._render_frame()

        info["landmark_images"] = landmark_images
        info["buildings_info"] = buildings_info

        return observation, info

    def step(self, actions: list) -> tuple:
        """Execute one step in the environment.
            
        Returns:
            observation: The current observation
            reward: The reward received
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        
        if self.start_time is None:
            self.start_time = time.perf_counter()
        self._agent_current_availability = self.multi_agent_controller.get_agent_availability()  
        # Find agents that were previously unavailable but are now available
        newly_available_agents = np.where((~self._agent_previous_availability) & self._agent_current_availability)[0]

        if self.debug:
            print(f"self._agent_previous_availability: {self._agent_previous_availability}")
            print(f"self._agent_current_availability: {self._agent_current_availability}")
            print(f"newly_available_agents: {newly_available_agents}")
        
        if len(newly_available_agents) != 0:
            agent_observation = [{}] * self.num_agents
            vision_input = self._get_obs(newly_available_agents)
            # print(f"vision_input: {vision_input}")
            self.multi_agent_controller.update_agent_transformation_hard(newly_available_agents)
            self._agent_location = self.multi_agent_controller.return_agent_location()
            self._agent_rotation = self.multi_agent_controller.return_agent_rotation()
            self.current_human_collision, self.current_object_collision, self.current_building_collision, self.current_vehicle_collision = self.multi_agent_controller.get_agent_collision(newly_available_agents)
            for i in newly_available_agents:
                vision = {}              
                if self.observation_type == 'rgb':
                    vision["egocentric view"] = {
                        "rgb": vision_input.get("rgb")[i]
                    }
                    self.agent_last_observation['rgb'][i] = vision_input.get("rgb")[i]
                elif self.observation_type == 'depth':
                    vision["egocentric view"] = {
                        "depth": vision_input.get("depth")[i]
                    }
                    self.agent_last_observation['depth'][i] = vision_input.get("depth")[i]
                elif self.observation_type == 'object_mask':
                    vision["egocentric view"] = {
                        "object_mask": vision_input.get("object_mask")[i]
                    }
                    self.agent_last_observation['object_mask'][i] = vision_input.get("object_mask")[i]
                elif self.observation_type == 'rgbd':
                    vision["egocentric view"] = {
                        "rgb": vision_input.get("rgb")[i],
                        "depth": vision_input.get("depth")[i]
                    }
                    self.agent_last_observation['rgb'][i] = vision_input.get("rgb")[i]
                    self.agent_last_observation['depth'][i] = vision_input.get("depth")[i]
                elif self.observation_type == 'all':
                    vision["egocentric view"] = {
                        "rgb": vision_input.get("rgb")[i],
                        "depth": vision_input.get("depth")[i],
                        "object_mask": vision_input.get("object_mask")[i]
                    }
                    self.agent_last_observation['rgb'][i] = vision_input.get("rgb")[i]
                    self.agent_last_observation['depth'][i] = vision_input.get("depth")[i]
                    self.agent_last_observation['object_mask'][i] = vision_input.get("object_mask")[i]
                agent_observation[i]["vision"] = vision
                agent_observation[i]["text"] = self.agent_current_messages[i]
                agent_observation[i]["orientation"] = self._transform_rotation(self._agent_rotation[i])
            
            self.multi_agent_controller.update_agent_observation(newly_available_agents, agent_observation)
            self.multi_agent_controller.update_agent_transformation_hard(newly_available_agents)
            self._agent_location = self.multi_agent_controller.return_agent_location()
            self._agent_rotation = self.multi_agent_controller.return_agent_rotation()
            self.current_human_collision, self.current_object_collision, self.current_building_collision, self.current_vehicle_collision = self.multi_agent_controller.get_agent_collision(newly_available_agents)

            # 更新移动距离
            if self._previous_agent_location is not None:
                self._moving_distance[newly_available_agents] += np.linalg.norm(self._agent_location[newly_available_agents][:2] - self._previous_agent_location[newly_available_agents][:2])
            
            # 更新前一个位置
            self._previous_agent_location = self._agent_location.copy()
            info = self._get_infor()
            self.multi_agent_controller.update_agent_availability(self._agent_current_availability)
            for i in newly_available_agents:
                self._write_trajectory(self.trajectory_writers[i], 
                                       self.steps_count[i], 
                                       self.agent_current_actions_start_time[i], 
                                       self.agent_current_actions[i], 
                                       self._agent_location[i], 
                                       self._agent_rotation[i], 
                                       self._target_location[i], 
                                       self._moving_distance[i], 
                                       self.current_human_collision[i], 
                                       self.current_object_collision[i], 
                                       self.current_building_collision[i],
                                       self.current_vehicle_collision[i],
                                       self.agent_current_messages[i],
                                       self.success)
            
            for i in newly_available_agents:
                self.agent_current_messages[i] = []
            
            if self.debug:
                print(f"agent_observation: {agent_observation}")
                print(f"observation: {len(observation)}")
                print(f"self.current_human_collision: {self.current_human_collision}")
                print(f"self.current_object_collision: {self.current_object_collision}")
                print(f"self._moving_distance: {self._moving_distance}")
            terminated = self._get_terminated()
            info = self._get_infor()
            rewards = self._get_reward()

        else:
            terminated = False
            self.current_human_collision, self.current_object_collision, self.current_building_collision, self.current_vehicle_collision = np.zeros(self.num_agents), np.zeros(self.num_agents), np.zeros(self.num_agents), np.zeros(self.num_agents)
            # observation = self._get_obs()
            info = self._get_infor()
            rewards = np.zeros(self.num_agents)
        observation = self.agent_last_observation
        # Update previous availability for next step
        self._agent_previous_availability = self._agent_current_availability.copy()

        # Execute action using buffer
        sent_actions, sent_actions_indexes, sent_messages = self.action_buffer.send_actions()
        if len(sent_actions) != 0:
            if self.debug:
                print(f"Sent actions: {sent_actions}")
                print(f"Sent actions indexes: {sent_actions_indexes}")
                print(f"Sent messages: {sent_messages}")
            action_start_time = time.perf_counter() - self.start_time
            
            for i, action in enumerate(sent_actions):
                match = re.search(r'Agent_(\d+)', action)
                if match:
                    if sent_actions_indexes[i] == 100:
                        if self._check_has_robot_in_view(observation["object_mask"][0]):
                            self.success = True
                        else:
                            self.success = False
                        terminated = True
                        for j in range(self.num_agents):
                            self._write_trajectory(self.trajectory_writers[j], 
                                                    self.steps_count[j], 
                                                    self.agent_current_actions_start_time[j], 
                                                    100, 
                                                    self._agent_location[j], 
                                                    self._agent_rotation[j], 
                                                    self._target_location[j], 
                                                    self._moving_distance[j], 
                                                    self.current_human_collision[j], 
                                                    self.current_object_collision[j], 
                                                    self.current_building_collision[j],
                                                    self.current_vehicle_collision[j],
                                                    self.agent_current_messages[j],
                                                    self.success)
                        break
                    else:
                        print(f"Agent {match.group(1)} action: {action}")
                        agent_index = int(match.group(1))-1
                        self._agent_previous_availability[agent_index] = False
                        self.agent_current_actions[agent_index] = sent_actions_indexes[i]
                        self.agent_current_actions_start_time[agent_index] = action_start_time
                        self.steps_count[agent_index] += 1

                        if len(sent_messages[i]) > 0:
                            message_target_agent = sent_messages[i]["target_agent"]
                            message_content = sent_messages[i]["content"]
                            self.agent_current_messages[message_target_agent].append({"source_agent": agent_index, "content": message_content})
            
            if self.debug:
                print(f"self.agent_current_actions: {self.agent_current_actions}")
                print(f"self.agent_current_actions_start_time: {self.agent_current_actions_start_time}")
                print(f"self.agent_current_messages: {self.agent_current_messages}")
                print(f"self.steps_count: {self.steps_count}")
            rewards = np.zeros(self.num_agents)


        # # Record video and save observations for each agent
        # current_time = time.perf_counter() - self.start_time
        # for i in range(self.num_agents):
        #     if self.record_video:
        #         # Save action video frames in actions subdirectory
        #         action_dir = os.path.join(self.images_dirs[i]['actions'], f"step_{self.steps_count:06d}")
        #         os.makedirs(action_dir, exist_ok=True)
        #         self.multi_agent_controller.render_video(current_time, self.record_video_fps, "lit", action_dir, agent_id=i)
            
        #     # Save observation image in observations subdirectory
        #     observation_path = os.path.join(
        #         self.images_dirs[i]['observations'], 
        #         f"step_{self.steps_count:06d}.png"
        #     )
        #     self.multi_agent_controller.get_image("lit", "file_path", observation_path, agent_id=i)

        if self.render_mode == "human":
            self._render_frame()
        # rewards = np.zeros(self.num_agents)
        return observation, rewards, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        img = self.multi_agent_controller.get_image("lit", "direct")
        if self.render_mode == "human":
            try:
                if 'get_ipython' in globals():
                    from IPython.display import display, Image
                    import cv2
                    # Convert to RGB for display
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Create a temporary file for display
                    temp_path = os.path.join(self.log_dir, "temp_display.png")
                    cv2.imwrite(temp_path, img_rgb)
                    display(Image(filename=temp_path))
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                else:
                    import cv2
                    cv2.imshow("CityNav Environment", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(1)
            except Exception as e:
                print(f"Warning: Could not render frame: {e}")
        return img
    
    def _write_trajectory(self, trajectory_writer, steps, action_start_time, action, agent_location, agent_rotation, target_location, moving_distance, human_collision, object_collision, building_collision, vehicle_collision, messages, success):
        if trajectory_writer is not None:
            action_meaning = self.action_meanings[str(action)]
            message_str = ""
            for message in messages:
                message_str += f"{message['source_agent']}: {message['content']}; "
            message_str = message_str[:-2] if message_str else ""  # Remove trailing "; "
            
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
                message_str,
                success
            ])
            trajectory_writer.flush()
    
    def close(self):
        if hasattr(self, 'trajectory_writers'):
            for writer in self.trajectory_writers.values():
                if writer is not None:
                    print(f"writer has been closed")
                    writer.flush()
                    writer.close()
            self.trajectory_writers = {}
        self.unrealcv.client.disconnect()