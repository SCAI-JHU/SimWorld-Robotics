import os
import cv2
import gym, simworld_gym
import numpy as np
import argparse
import warnings

from utils import numpy_to_base64, split_into_strips, action_history_text, save_images, log_lines
from agents import ReasoningAgent, ReActAgent
from prompt_template import nav_template, reasoning_template, perception_template

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run the agent in a simulated environment.")
parser.add_argument("--map", type=str, default="20_0", help="Number of the World.")
parser.add_argument("--backend", type=str, default="openai", help="Backend for the LLM.")
parser.add_argument("--model", type=str, default="gpt-4o", help="Model name.")
parser.add_argument("--ip", type=str, default="127.0.0.1")
parser.add_argument("--reasoning", action="store_true", help="Use ReasoningAgent (single-step plan) instead of ReAct.")
parser.add_argument("--setting", type=str, choices=["simple", "traffic"], default="simple", help="Task setting: simple or traffic.")
parser.add_argument("--strip", action="store_true", help="Use strip mode for perception.")
parser.add_argument("--depth", action="store_true", help="Use depth mode for perception.")
parser.add_argument("--segment", action="store_true", help="Use segmentation mode for perception.")

args = parser.parse_args()
map = args.map
backend = args.backend
ip = args.ip
setting = args.setting
model = args.model
reasoning = args.reasoning
log_dir = f"{model}_{setting}"
strip = args.strip
depth = args.depth
segment = args.segment

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

ue_port = int(os.getenv("UE_PORT"))

print(f"Map: {map}, Backend: {backend}, Model: {model}, UE Port: {ue_port}")

reward_setting = {
    "human_collision_penalty": -1,
    "object_collision_penalty": -0.1,
    "action_penalty": -0.1,
    "success_reward": 10.0,
    "off_track_penalty": -0.01
}
env = gym.make(
    'gym_citynav/SimpleWorld-v0' if setting == "simple" else 'gym_citynav/TrafficWorld-v0',
    port=ue_port,
    resolution=(720, 600),
    render_mode="rgb_array",
    observation_type="all",
    record_video=False,
    log_dir=log_dir,
    reward_setting=reward_setting,
)

# Initialize agent based on mode
if reasoning:
    agent = ReasoningAgent(
        backend=backend,
        model=model,
        system_prompt=nav_template(strip=strip, depth=depth, segment=segment),
    )
else:
    agent = ReActAgent(
        backend=backend,
        model=model,
        reasoning_prompt=reasoning_template(strip=strip),
        perception_prompt=perception_template(strip=strip, depth=depth, segment=segment),
    )

initialized = False

action_mapping = ["Move_Forward", "Rotate_Left", "Rotate_Right", "Move_Left", "Move_Right", "Subtask_Completed"]
task = 11 if int(map.split("_")[1]) < 50 else 21
task_2_test = [f"task_dist_{task}_0_1"]

input_tokens = 0
output_tokens = 0

for task in task_2_test:
    task_path = os.path.join("single_agent_world", "easy", f"map_road_{map}", task)
    world_json = os.path.join(task_path, "progen_world.json")
    agent_json = os.path.join(task_path, "task_config.json")
    if not initialized:
        options = {
            "task_path": task_path,
            "agent_json": agent_json,
            "world_json": world_json,
        }
        if setting != "simple":
            traffic_json = os.path.join(task_path, "traffic.json")
            options["traffic_json"] = traffic_json
        initialized = True
    else:
        options = {
            "task_path": task_path,
            "agent_json": agent_json,
        }
        if setting != "simple":
            traffic_json = os.path.join(task_path, "traffic.json")
            options["traffic_json"] = traffic_json
    observation, info = env.reset(options=options)
    vision_cue = info["current_instruction"]["image"]
    instruction = info["current_instruction"]["text"]
    action_history = []
    chosen_actions = []
    
    # Reset agent state for new task
    agent.reset_state()
    
    folder_path = os.path.join("/SimWorld", "agent_log", f"{log_dir}", f"{map}", f"{task}")
    os.makedirs(folder_path , exist_ok=True)

    i = 0
    terminated = False
    last_position = None
    current_position = None
    parse_failure_count = 0
    ACTION_TO_STEP = {0: 0, 1: 5, 2: 4, 3: 2, 4: 3}

    while True:
        orientation = info['agent']['agent_rotation']
        current_position = info['agent']['agent_location']
        forward_count = sum(1 for a in chosen_actions if a == 0)
        if not last_position is None and forward_count > 3 and np.linalg.norm(np.array(current_position) - np.array(last_position)) < 0.5:
            print("Stuck in place, terminating.")
            log_lines(folder_path, [("STUCK", "")])
            break
        last_position = current_position
        save_images(
            observation["rgb"], vision_cue,
            os.path.join(folder_path, f"display_{i}.png")
        )

        # Preprocess images
        images = []
        if strip:
            strips = split_into_strips(observation["rgb"])
            for idx, img in enumerate(strips):
                images.append({"img": numpy_to_base64(img), "description": [
                    'The view on the left',
                    'The horizontal center',
                    'The right',
                ][idx] if idx < 3 else None})
            images.append({"img": numpy_to_base64(vision_cue), "description": 'The expected view'})
        else:
            images.append({"img": numpy_to_base64(observation["rgb"]), "description": 'The current view'})
            images.append({"img": numpy_to_base64(vision_cue), "description": 'The expected view'})
        if segment:
            images.append({"img": numpy_to_base64(observation["object_mask"]), "description": 'The object segmentation mask of the current view'})
        if depth:
            images.append({"img": numpy_to_base64(observation["depth_map"]), "description": 'The depth map of the current view'})
        
        # Preprocess action history text
        action_history_str = action_history_text(action_history, action_mapping)
        
        try:
            result = agent.step(
                observation=images,
                instruction=instruction,
                orientation=orientation,
                action_history_text=action_history_str,
                chosen_actions=chosen_actions,
            )
            chosen_actions = result.get("actions", None)
            vision_description = result.get("vision_description", "")
            summary = result.get("summary", "")
            match = result.get("match", None)
            reason = result.get("reason", "")
            usage = result.get("usage", {"input": 0, "output": 0})
            input_tokens += usage.get("input", 0)
            output_tokens += usage.get("output", 0)
            print(f"[vision {i}]", vision_description)
            if reason:
                print(f"[reason {i}]", reason)
            log_entries = [
                ("current subtask", instruction),
                (f"vision {i}", str(vision_description).replace("\n", "")),
                (f"summary {i}", str(summary).replace("\n", "")),
                (f"actions {i}", str(chosen_actions)),
            ]
            if reason:
                log_entries.insert(2, (f"reason {i}", str(reason).replace("\n", "")))
            if match is not None:
                log_entries.append((f"match {i}", str(match)))
            log_lines(folder_path, log_entries)
        except Exception as e:
            print(f"Agent step error: {e}")
            parse_failure_count += 1
            if parse_failure_count > 10:
                print("Too many errors, terminating.")
                log_lines(folder_path, [("TOO MANY ERRORS", "")])
                break
            continue
        if not chosen_actions:
            print("No action specified. Failing.")
            break
        for chosen_action in chosen_actions:
            action_history.append(chosen_action)
            i += 1
            if chosen_action == -1:
                observation, _, terminated, _, info = env.step(-1)
                if terminated:
                    print("End")
                    break
                instruction = info["current_instruction"]["text"]
                vision_cue = info["current_instruction"]["image"]
                action_history = []
            else:
                step_code = ACTION_TO_STEP.get(chosen_action)
                if step_code is None:
                    continue
                observation, _, terminated, _, info = env.step(step_code)
        if terminated:
            print("End")
            break
                    
env.close()