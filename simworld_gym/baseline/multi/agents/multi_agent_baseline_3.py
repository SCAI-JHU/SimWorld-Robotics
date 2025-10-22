import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gpt4o_wrapper import GPT4oVision
from simworld_gym import Vector, Edge, Node
from multi_utils import extract_img_from_obs, safe_json_from_llm, save_img
from route_instruction_generator import RouteInstructionGenerator
from base_agent_buffer import BaseAgentBuffer


class MultiAgentFirstBaselineAgent(BaseAgentBuffer):
    """Multi-agent baseline: Agent 1 navigates with LLM in 2D city grid, Agent 0 does robot detection."""
    
    NOOP_ACTION = 0
    def __init__(self, action_space, action_config, action_buffer,
                 agent_controller, num_agents, self_index,
                 landmark_images, buildings_info):
        super().__init__(action_space, action_config, action_buffer, agent_controller)
        self.num_agents = num_agents
        self.self_index = self_index
        self.instruction: Optional[List[str]] = None
        self.is_finished = False
        self.debug = False
        
        self.route_instruction_generator = RouteInstructionGenerator()
        self.landmark_images = landmark_images
        self.buildings_info = buildings_info
        self.gpt = GPT4oVision()
        
        self.sub_instr: List[str] = []
        self.cur_sub_idx = 0
        self.memory_summary = ""
        self.action_trace: List[int] = []
        self.VALID_ACTIONS = list(range(self.action_space.n)) + [-1] + [100]
        
        self.save_dir = f"debug_imgs/agent_{self.self_index}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.frame_idx = 0
        
    def set_instruction(self, instr: List[str]):
        """Set navigation instructions for the agent."""
        if not (isinstance(instr, list) and all(isinstance(s, str) for s in instr)):
            print(f"[Agent-{self.self_index}] Invalid instruction format: {instr}")
            return
        
        self.instruction = instr
        self.sub_instr = instr
        self.cur_sub_idx = 0
        self.memory_summary = ""
        self.action_trace.clear()
        self.is_finished = False

    def act(self, debug=False):
        """Main action loop for the agent."""
        self.debug = debug
        while not self.is_finished:
            if self.agent_controller.return_agent_availability():
                obs = self.agent_controller.get_observation()
                action_id, message = self._generate_action(obs)
                print(message)
                buf = self.agent_controller.interpret_action_to_buffer_action(action_id, message)
                self.send_action_to_buffer(buf)
                self.agent_controller.update_agent_availability(False)

                if self.debug:
                    print(f"[Agent-{self.self_index}] sent action {action_id}")
                time.sleep(0.5)
            time.sleep(0.1)

    def _generate_action(self, obs):
        """Generate action based on observation."""
        if self.self_index == 1:
            if self.cur_sub_idx >= len(self.sub_instr):
                self.is_finished = True
                return self.NOOP_ACTION, {"done"}

            cur_sub = self.sub_instr[self.cur_sub_idx]
            img = extract_img_from_obs(obs)

            if img is None:
                if self.debug:
                    print(f"[Agent-{self.self_index}] No image frame in observation")
                return self.NOOP_ACTION, {}
                
            save_img(img, self.save_dir, self.frame_idx, self.self_index)
            self.frame_idx += 1

            sys_prompt = (
                "You are a navigation robot in a 2D city grid.\n"
                "Primitive actions: 0=forward, 1=backward, 4=turn-right, 5=turn-left, -1=done, 100=see-robot dog.\n"
                "Never output -100. Only these integers are allowed: 0,1,4,5,-1,100.\n"
                "Return ONLY a JSON object:\n"
                "{\"Summary\":\"<updated history>\",\"Reason\":\"<why>\",\"Action\":<int>}\n"
                "Don't output anything else."
            )
            
            orientation = obs.get("orientation")
            if orientation is not None:
                self.orientation = orientation
                if self.debug:
                    print(f"Agent {self.agent_controller.agent_name} Get Orientation: {self.orientation}")
                    
            usr_prompt = (
                f"Current sub-instruction: \"{cur_sub}\"\n"
                f"Actions taken so far (this sub): {self.action_trace}\n"
                f"History summary: {self.memory_summary}\n"
                f"Current orientation: {orientation}\n"
                "Please consider the current orientation when making decisions.\n"
                "What's the next primitive action?"
            )
            
            raw = self.gpt.chat(sys_prompt, usr_prompt, image=img, max_tokens=200)
            print(cur_sub)
            print(f"[Agent-{self.self_index+1}] raw response: {raw}")
            
            obj = safe_json_from_llm(raw) or {}
            msg = obj
            act = int(obj.get("Action", 0))

            print("agent2_act:", act)
            if act not in self.VALID_ACTIONS:
                print("[GPT] Invalid action, fallback to 0")
                act = 0

            self.memory_summary = obj.get("Summary", self.memory_summary)
            self.action_trace.append(act)

            if act == -1:
                print("Finished current sub_instruction:", self.sub_instr[self.cur_sub_idx])
                self.cur_sub_idx += 1
                self.action_trace.clear()
                act = 0
                if self.cur_sub_idx >= len(self.sub_instr):
                    print(f"[Agent-{self.self_index}] All instructions done")
                    self.is_finished = True
            return act, {}
        else:
            vision = obs["vision"]
            egocentric_view = vision.get("egocentric view")
            destination_view = vision.get("destination view")
            
            if destination_view is not None:
                self.destination_view = destination_view
                if self.debug:
                    print(f"Agent {self.agent_controller.agent_name} Get Destination View")
                    
            if egocentric_view is not None:
                self.rgb_view = egocentric_view.get("rgb")
                self.depth_view = egocentric_view.get("depth")
                self.object_mask_view = egocentric_view.get("object_mask")
                if self.debug:
                    print(f"Agent {self.agent_controller.agent_name} Get Egocentric View")
                    
            orientation = obs.get("orientation")
            if orientation is not None:
                self.orientation = orientation
                if self.debug:
                    print(f"Agent {self.agent_controller.agent_name} Get Orientation: {self.orientation}")
                    
            messages = obs["text"]
            for message in messages:
                if message["source_agent"] == "instruction":
                    self.instruction = message["content"]
                if self.debug:
                    print(f"Agent {self.agent_controller.agent_name} Get Message from Agent {message['source_agent']}: {message['content']}")
            
            action = 5
            img = extract_img_from_obs(obs)
            sys_prompt = (
                "You are a robot vision analyzer. "
                "Your job is to detect if a robot appears in the given image.\n"
                "Return ONLY an integer: 100 if you see a robot, -100 otherwise."
            )
            usr_prompt = "Please analyze the current view and decide whether a robot is visible."
            
            if self.ground_truth_action is not None and len(self.ground_truth_action) > 0:
                action = self.ground_truth_action.pop(0)
                print("agent1_act:", action)
            else:
                is_done = self.gpt.chat(sys_prompt, usr_prompt, image=img, max_tokens=200)
                print("Is_done:", is_done)
                if is_done.strip() == "100":
                    action = 100
                elif is_done.strip() == "-100":
                    action = -100
                    
            message = {"target_agent": self.num_agents - self.self_index - 1, "content": "I want to break free"}
            return action, message
        
    def load_env(self, city_path: str, road_path: str):
        self.route_instruction_generator.load_env(city_path, road_path)

    def generate_instructions(self, start: List[float], end: List[float],
                              output_path: str, if_count_intersections: bool = True) -> str:
        s, e = Node(Vector(*start), "normal"), Node(Vector(*end), "normal")
        self.route_instruction_generator.updated_map = self._update_map(s, e)
        return self.route_instruction_generator.generate_instructions(
            s, e, output_path, if_count_intersections
        )

    def _update_map(self, s: Node, e: Node):
        near_s = self.find_nearest_n_nodes(s, 2)
        near_e = self.find_nearest_n_nodes(e, 2)
        m = self.route_instruction_generator._load_strctured_map()
        m.add_node(s); m.add_node(e)
        m.add_edge(Edge(near_s[0][0], s)); m.add_edge(Edge(s, near_s[1][0]))
        m.add_edge(Edge(near_e[1][0], e)); m.add_edge(Edge(e, near_e[0][0]))
        return m

    def find_nearest_n_nodes(self, node: Node, n=2):
        tgt = node.position
        nodes = self.route_instruction_generator.basic_map.nodes
        d = [(nd, nd.position.distance(tgt)) for nd in nodes]
        return sorted(d, key=lambda x: x[1])[:n]
    
    def rule_based_navigation_actions(self, start: List[float], end: List[float], start_orientation: float = 0):
        start_node = Node(Vector(start[0], start[1]), "normal") 
        end_node = Node(Vector(end[0], end[1]), "normal")
        self.route_instruction_generator.updated_map = self._update_map(start_node, end_node)
        actions = self.route_instruction_generator.generate_oracle_path(start_node, end_node, start_orientation)
        return actions
    
    def update_ground_truth_action(self, actions: List[Node]):
        self.ground_truth_action = actions
    
