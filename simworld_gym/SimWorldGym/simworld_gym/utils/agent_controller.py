import numpy as np
import time
import os
from simworld_gym.utils import misc
from simworld_gym.utils.base import Action
from simworld_gym.utils.unrealcv_basic import UnrealCV

AGENT_ASSET_PATH = "asset/AgentPath.json"

class AgentController:
    def __init__(self, client: UnrealCV, resolution):
        self.client = client
        self.resolution = resolution
        self.agent_name = None
        self.agent_setting = None
        self._agent_location = None
        self._agent_rotation = None
        self._target_location = None
        self._instruction = None
        self._instruction_image_path = None
        self._evaluation_of_subtask = None
        self.camera_id = None
        self.agent_library = misc.load_config(AGENT_ASSET_PATH)
        self.action_config = None
        self.availability = False
        self.observation = {}
        self.initial_camera_setting = True
        self.reset_time = 0

    def reset(self, agent_json=None, if_single_agent=True):
        if agent_json:
            self.update_json(agent_json)
        if self.reset_time == 0:
            self._generate_agent()
        else:
            location = self.agent_setting.get("spawning_point", {}).get("location", {"x": 0, "y": 0, "z": 0})
            print(f"spawning_point: {location}")
            location = [location.get("x", 0), location.get("y", 0), location.get("z", 0)]

            orientation = self.agent_setting.get("spawning_point", {}).get("orientation", {"roll": 0, "pitch": 0, "yaw": 0})
            orientation = [orientation.get("roll", 0), orientation.get("pitch", 0), orientation.get("yaw", 0)]
            self._set_agent_location(location)
            self._set_agent_rotation(orientation)
        # Ensure target-related fields are populated every reset
        try:
            self._read_target_info(if_single_agent=if_single_agent)
        except Exception:
            # Keep reset robust even if target info is temporarily unavailable
            pass
        self.reset_time += 1

    def update_json(self, agent_json):
        try:
            self.agent_setting = agent_json
            self.agent_name = self.agent_setting.get("agent_name", "default_agent_name")
        except Exception as e:
            print(f"Error updating agent settings: {e}")

    def _generate_agent(self):
        try:
            location = self.agent_setting.get("spawning_point", {}).get("location", {"x": 0, "y": 0, "z": 0})
            print(f"spawning_point: {location}")
            location = [location.get("x", 0), location.get("y", 0), location.get("z", 0)]

            orientation = self.agent_setting.get("spawning_point", {}).get("orientation", {"roll": 0, "pitch": 0, "yaw": 0})
            orientation = [orientation.get("roll", 0), orientation.get("pitch", 0), orientation.get("yaw", 0)]

            instance_name = self.agent_setting.get("instance_name")
            instance_ref = self.agent_library.get(instance_name).get("asset_path")
            agent_type = self.agent_library.get(instance_name).get("color")
            agent_color = self._parse_rgb(self.agent_library.get("color").get(agent_type))

            if not instance_ref:
                raise ValueError(f"Agent instance {instance_name} not found in library.")

            self._spawn_agent(instance_ref, agent_color)
            self._enable_agent_controller(True)
            self._set_agent_location(location)
            self._set_agent_rotation(orientation)
            self.update_agent_transformation_hard()
            self._set_camera_id()
            self.client.client.request("vset /camera/{}/size {} {}".format(self.camera_id, self.resolution[0], self.resolution[1]))
            print(f"Setting up camera {self.camera_id}")
            self.client.client.request("vset /camera/{}/reflection Lumen".format(self.camera_id))
            self.client.client.request("vset /camera/{}/illumination Lumen".format(self.camera_id))
            self.client.client.request("vset /camera/{}/fov 120".format(self.camera_id))
        except Exception as e:
            print(f"Error generating robot: {e}")

    def _spawn_agent(self, agent_model, agent_color=[0,0,0]):
        try:
            self.client.spawn_bp_asset(agent_model, self.agent_name)
            self.client.set_color(self.agent_name, agent_color)
        except Exception as e:
            print(f"Error spawning agent {self.agent_name}: {e}")

    def _enable_agent_controller(self, is_enable):
        try:
            self.client.enable_controller(self.agent_name, is_enable)
        except Exception as e:
            print(f"Error enabling controller for {self.agent_name}: {e}")

    def _set_agent_location(self, location):
        try:
            if self.agent_name:
                self.client.set_location(location, self.agent_name)
            else:
                print("Invalid agent name.")
        except Exception as e:
            print(f"Error setting location for {self.agent_name}: {e}")

    def _set_agent_rotation(self, rotation):
        """Set the rotation/orientation of the agent in the environment.
        
        Args:
            rotation (list): A list of 3 float values representing [roll, pitch, yaw] in degrees.
                           roll: Rotation around X axis
                           pitch: Rotation around Y axis 
                           yaw: Rotation around Z axis

        The method uses the UnrealCV client to set the agent's orientation. If there's an error,
        it will catch and print the exception message.
        """
        try:
            self.client.set_orientation(rotation, self.agent_name)
        except Exception as e:
            print(f"Error setting rotation for {self.agent_name}: {e}")

    def update_agent_transformation(self, location, rotation):
        self._agent_location = location
        self._agent_rotation = rotation
    
    def update_agent_transformation_hard(self):
        self._agent_location = self.get_agent_location_hard()
        self._agent_rotation = self.get_agent_rotation_hard()
    
    def update_agent_availability(self, is_available):
        self.availability = is_available

    def update_agent_availability_hard(self):
        self.availability = self.client.get_is_available(self.agent_name)

    def update_observation(self, observation):
        self.observation = observation
    
    def get_observation(self):
        return self.observation
    
    def reset_observation(self):
        self.observation = {}
    
    def get_agent_rotation_hard(self):
        try:

            return self.client.get_orientation(self.agent_name)
        except Exception as e:
            print(f"Error getting rotation for {self.agent_name}: {e}")
            return None

    def get_agent_collision(self):
        try:
            return self.client.get_total_collision(self.agent_name)
        except Exception as e:
            print(f"Error getting collision for {self.agent_name}: {e}")
            return None
    
    def _set_camera_id(self):
        # self.camera_id = len(self._get_cameras())-1
        self.camera_id = self.agent_setting.get("camera_id", -1)
        print(f"Agent's Camera ID: {self.camera_id}")

    def get_agent_location_hard(self):
        try:
            return self.client.get_location(self.agent_name)
        except Exception as e:
            print(f"Error getting location for {self.agent_name}: {e}")
            return None

    def _destroy_agent(self):
        try:
            if self.agent_name:
                self.client.destroy_hard(self.agent_name)
                print(f"Successfully destroyed agent {self.agent_name}")
                self.client.clean_garbage()
                print(f"Cameras: {self.client.get_cameras()}")
        except Exception as e:
            print(f"Error destroying agent {self.agent_name}: {e}")

    def apply_action(self, action):
        """Apply an action to the agent.
        
        Takes a discrete action index and executes the corresponding movement or rotation
        based on the action_config. The action is parsed from the action_meanings mapping
        and the parameters are retrieved using _get_action_params.

        For movement actions (Move_Speed), calls _apply_action_transition().
        For rotation actions (Rotate_Angle), calls _apply_action_rotation().

        Args:
            action (int): The discrete action index to execute

        Returns:
            float: The duration parameter of the executed action
            
        Raises:
            Exception: If there is an error applying the action
        """
        try:
            # Parse discrete action based on action_meanings
            action_meaning = self.action_config["action_meanings"][str(action)]
            
            # Get action parameters using helper method
            action_type, params = self._get_action_params(action_meaning)
            
            if action_type == "Move_Speed":
                self._apply_action_transition(params)
                return params[1]
            elif action_type == "Rotate_Angle":
                self._apply_action_rotation(params)
                return params[0]
            print(f"Action {action_meaning} applied with parameters {params}")

        except Exception as e:
            print(f"Error applying action {action}: {e}")
            return 0

    def interpret_action_to_buffer_action(self, action, message={}):
        """Convert action index to buffer action.
        
        Args:
            action (int): Action index
            message (dict): Message to be sent to the buffer
            
        Returns:
            Action: Buffer action object
        """
        try:
            if action == None:
                if len(message) > 0:
                    return Action(self.agent_name, "send message", [], action_index=6, message=message)
                else:
                    return None
            else:
                action_meaning = self.action_config["action_meanings"][str(action)]
                action_type, params = self._get_action_params(action_meaning)
                if len(message) > 0:
                    return Action(self.agent_name, action_type, params, action_index=action, message=message)
                else:
                    return Action(self.agent_name, action_type, params, action_index=action)
            
        except Exception as e:
            print(f"Error interpreting action {action}: {e}")
            return None

    def _apply_action_transition(self, action):
        try:
            self.client.apply_action_transition(self.agent_name, action)
        except Exception as e:
            print(f"Error applying transition action to {self.agent_name}: {e}")

    def _apply_action_rotation(self, action):   
        try:
            self.client.apply_action_rotation(self.agent_name, action)
        except Exception as e:
            print(f"Error applying rotation action to {self.agent_name}: {e}")

    def get_image(self, view_mode, mode, file_path=None):
        try:
            self.client.client.request("vset /camera/{}/exposure_bias 0".format(self.camera_id))
            self.client.client.request("vset /camera/{}/exposure_bias 4.5".format(self.camera_id))
            if file_path:
                img = self.client.read_image(self.camera_id, view_mode, mode, file_path)
            else:
                img = self.client.read_image(self.camera_id, view_mode, mode)
            self.client.client.request("vset /camera/{}/exposure_bias 0".format(self.camera_id))
            return img
        except Exception as e:
            print(f"Error getting image from camera {self.camera_id}: {e}")
            return None
    
    def render_video(self, duration, fps, render_mode, image_path):
        try:
            start_time = time.perf_counter()
            interval = 1/fps
            frame_idx = 0
            while time.perf_counter() - start_time < duration:
                file_name = os.path.join(image_path, f"frame_{frame_idx:04d}.png")
                request_start = time.perf_counter()
                self.get_image(render_mode, "file_path", file_name)
                request_time = time.perf_counter() - request_start
                sleep_time = max(0, interval - request_time)  # 调整 sleep 计算，确保节奏稳定
                time.sleep(sleep_time)
                frame_idx += 1
        except Exception as e:
            print(f"Error rendering video: {e}")

    def _get_cameras(self):
        try:
            return self.client.get_cameras()
        except Exception as e:
            print(f"Error getting cameras in the environment : {e}")
            return None

    def _read_target_info(self, if_single_agent=True):
        self._set_target_location()
        if if_single_agent:
            self._read_instruction()
            self._read_instruction_image_path()
            self._read_evaluation_of_subtask()
    
    def _parse_rgb(self, color_str):
        """Parse RGB values from color string like '(R=255,G=255,B=0)'"""
        import re
        pattern = r'R=(\d+),G=(\d+),B=(\d+)'
        match = re.search(pattern, color_str)
        if match:
            return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
        return [0, 0, 0]  # Default to black if parsing fails

    def _set_target_location(self):
        self._target_location = np.array(
            [self.agent_setting["destination"]["x"], self.agent_setting["destination"]["y"],
             self.agent_setting["destination"]["z"]])

    def _read_instruction(self):
        self._instruction = self.agent_setting["instruction"]

    def _read_instruction_image_path(self):
        self._instruction_image_path = self.agent_setting["robot_view_images"]
    
    def _read_evaluation_of_subtask(self):
        self._evaluation_of_subtask = self.agent_setting["evaluation_of_subtask"]
    
    def _get_action_params(self, action_meaning):
        """Extract action parameters based on action meaning.
        
        Args:
            action_meaning (str): The meaning of the action (e.g., "MOVE_FORWARD", "TURN_RIGHT")
            
        Returns:
            tuple: (action_type, params) where action_type is str and params is list
        """
        try:
            if action_meaning in ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "MOVE_RIGHT"]:
                params = self.action_config["action_parameters"]["movement"]
                return "Move_Speed", [
                    params["speed"],
                    params["duration"],
                    params["directions"][action_meaning]["direction"]
                ]
            
            elif action_meaning in ["TURN_RIGHT", "TURN_LEFT"]:
                params = self.action_config["action_parameters"]["rotation"]
                return "Rotate_Angle", [
                    params["duration"],
                    params["angle"][action_meaning],
                    params["directions"][action_meaning]["direction"]
                ]
            
            elif action_meaning == "TERMINATE":
                return "Terminate", []
            
            raise ValueError(f"Unknown action meaning: {action_meaning}")
            
        except Exception as e:
            raise Exception(f"Error getting action parameters: {e}")

    def return_agent_location(self):
        return self._agent_location

    def return_agent_rotation(self):
        return self._agent_rotation

    def return_target_location(self):
        return self._target_location

    def return_criteria(self):
        return self.agent_setting["criteria"]
    
    def return_agent_availability(self):
        return self.availability
    
    def return_instruction(self):
        if self._instruction is None and self.agent_setting is not None:
            try:
                self._read_instruction()
            except Exception:
                pass
        return self._instruction
    
    def return_instruction_image_path(self):
        if self._instruction_image_path is None and self.agent_setting is not None:
            try:
                self._read_instruction_image_path()
            except Exception:
                pass
        return self._instruction_image_path
    
    def return_evaluation_of_subtask(self):
        if self._evaluation_of_subtask is None and self.agent_setting is not None:
            try:
                self._read_evaluation_of_subtask()
            except Exception:
                pass
        return self._evaluation_of_subtask


