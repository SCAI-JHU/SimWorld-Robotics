from simworld_gym.utils.agent_controller import AgentController
import numpy as np


class MultiAgentController:
    def __init__(self, client, resolution):
        self.client = client
        self.resolution = resolution
        self.action_config = None
        self.agent_controllers = []
        self.action_buffer = None
        self.agents_name = []
        self._agent_name_cache = ()
        self.reset_time = 0

    def _valid_agent_indexes(self, agent_indexes):
        limit = len(self.agent_controllers)
        return [idx for idx in agent_indexes if 0 <= idx < limit]

    def reset(self, agent_json=None):
        agents = agent_json.get("agents", []) if agent_json else []
        if self.reset_time == 0:
            self.agents_name = []
            self.agent_controllers = []
            for agent_config in agents:
                controller = AgentController(self.client, self.resolution)
                single_agent_config = {
                    "criteria": agent_json.get("criteria", {}),
                    **agent_config,
                }
                agent_name = agent_config["agent_name"]
                self.agents_name.append(agent_name)
                controller.reset(single_agent_config, if_single_agent=False)
                controller.action_config = self.action_config
                self.agent_controllers.append(controller)

            self._agent_name_cache = tuple(self.agents_name)
            if agents:
                self.client.clean_garbage()
        else:
            for i, agent_config in enumerate(agents):
                if i >= len(self.agent_controllers):
                    break
                spawn = agent_config.get("spawning_point", {})
                location_cfg = spawn.get("location", {})
                orientation_cfg = spawn.get("orientation", {})
                print(f"spawning_point: {location_cfg}")

                location = [
                    location_cfg.get("x", 0),
                    location_cfg.get("y", 0),
                    location_cfg.get("z", 0),
                ]
                orientation = [
                    orientation_cfg.get("roll", 0),
                    orientation_cfg.get("pitch", 0),
                    orientation_cfg.get("yaw", 0),
                ]

                controller = self.agent_controllers[i]
                controller._set_agent_location(location)
                controller._set_agent_rotation(orientation)

        self.reset_time += 1

    def interpret_action_to_buffer_action(self, actions):
        buffer_actions = []
        for i, action in enumerate(actions):
            if action:
                buffer_actions.append(self.agent_controllers[i].interpret_action_to_buffer_action(action))
        return buffer_actions

    def update_agent_transformation_hard(self, agent_indexes):
        valid_indexes = self._valid_agent_indexes(agent_indexes)
        if not valid_indexes:
            return

        agent_names = [self.agents_name[i] for i in valid_indexes]
        locations = self.client.get_location_batch(agent_names)
        rotations = self.client.get_orientation_batch(agent_names)
        for offset, idx in enumerate(valid_indexes):
            self.agent_controllers[idx].update_agent_transformation(locations[offset], rotations[offset])

    def get_image(self, agent_indexes, view_mode, mode, file_path=None):
        images = [None] * len(self.agent_controllers)
        for idx in self._valid_agent_indexes(agent_indexes):
            images[idx] = self.agent_controllers[idx].get_image(view_mode, mode, file_path)
        return images

    def get_agent_collision(self, agent_indexes):
        size = len(self.agent_controllers)
        human = np.zeros(size)
        obj = np.zeros(size)
        building = np.zeros(size)
        vehicle = np.zeros(size)

        valid_indexes = self._valid_agent_indexes(agent_indexes)
        if not valid_indexes:
            return human, obj, building, vehicle

        agent_names = [self.agents_name[i] for i in valid_indexes]
        collisions = self.client.get_total_collision_batch(agent_names)

        for (h, o, b, v), idx in zip(collisions, valid_indexes):
            human[idx] = h
            obj[idx] = o
            building[idx] = b
            vehicle[idx] = v

        return human, obj, building, vehicle

    def get_agent_availability(self):
        if not self.agents_name:
            return np.empty(0, dtype=bool)
        names = self._agent_name_cache or tuple(self.agents_name)
        is_available = self.client.get_available_batch(names)
        return np.array(is_available, dtype=bool)

    def update_agent_availability(self, is_available):
        for controller, available in zip(self.agent_controllers, is_available):
            controller.update_agent_availability(available)

    def update_agent_observation(self, agent_indexes, observation):
        for idx in self._valid_agent_indexes(agent_indexes):
            self.agent_controllers[idx].update_observation(observation[idx])

    def return_agent_location(self):
        if not self.agent_controllers:
            return np.empty((0, 3))
        return np.asarray([controller.return_agent_location() for controller in self.agent_controllers])

    def return_agent_rotation(self):
        if not self.agent_controllers:
            return np.empty((0, 3))
        return np.asarray([controller.return_agent_rotation() for controller in self.agent_controllers])

    def return_target_location(self):
        if not self.agent_controllers:
            return np.empty((0, 3))
        return np.asarray([controller.return_target_location() for controller in self.agent_controllers])

    def return_criteria(self):
        return self.agent_controllers[0].return_criteria() if self.agent_controllers else None

    def return_agent_availability(self):
        return [controller.return_agent_availability() for controller in self.agent_controllers]

    def return_instruction(self):
        return [[{"source_agent": "instruction", "content": controller.return_instruction()}] for controller in self.agent_controllers]

    def return_instruction_image_path(self):
        return [controller.return_instruction_image_path() for controller in self.agent_controllers]
