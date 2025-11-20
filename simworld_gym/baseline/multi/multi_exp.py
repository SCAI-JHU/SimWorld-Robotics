#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import threading
import warnings
import argparse
from typing import Tuple, Dict

import gym
import numpy as np

# env / plotting backends
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from simworld_gym.utils import misc
from multi_utils import (
    get_pos,
    describe_image,
    load_dataset_from_info,
    tournament_winner,
    get_closest_building_by_midpoint,
    get_closest_building_by_location,
)


# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(description="Unified multi-agent experiments (settings 1, 2, 3)")
parser.add_argument("--setting", type=int, choices=[1, 2, 3], default=1,
                    help="Which experiment to run: 1 = LLM image-match target; 2 = LLM + rendezvous; 3 = fixed target (id=13) + rendezvous")
parser.add_argument("--map", type=str, default="20", help="Map id (string)")
parser.add_argument("--task", type=int, default=1, help="Task id number")
parser.add_argument("--suffix", type=str, default=None, help="Task suffix (string); defaults depend on setting")
parser.add_argument("--port", type=int, default=int(os.getenv("UE_PORT", 9900)), help="UnrealCV port")
parser.add_argument("--log_dir", type=str, default="openai_multi_setting_1", help="Directory to save env logs")
parser.add_argument("--episodes", type=int, default=1, help="How many episodes to run")
parser.add_argument("--max_steps", type=int, default=100, help="Max steps per episode (0 = unlimited)")
parser.add_argument("--debug", action="store_true", help="Enable gym_citynav debug mode")

# LLM params (used by setting 1 and 2; 3 also calls describe_image but doesn't rely on tournament)
parser.add_argument("--backend", type=str, default="openai", help="LLM backend: openai | gemini | other")
parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model name")
parser.add_argument("--fixed_target_id", type=int, default=13, help="Setting 3: fixed target building id")

args = parser.parse_args()


# -----------------------------
# Dynamic imports per setting
# -----------------------------
if args.setting == 1:
    from ue_agent.multi_agent_baseline_1 import MultiAgentFirstBaselineAgent as AgentClass
elif args.setting == 2:
    from ue_agent.multi_agent_baseline_2 import MultiAgentFirstBaselineAgent as AgentClass
else:  # 3
    from ue_agent.multi_agent_baseline_3 import MultiAgentFirstBaselineAgent as AgentClass


# -----------------------------
# LLM client factory (sync, used by multi_utils)
# -----------------------------
def build_llm_client(backend: str, model: str):
    from openai import OpenAI
    if backend == "openai":
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY")), model
    if backend == "gemini":
        return OpenAI(api_key=os.environ.get("GEMINI_API_KEY"),
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/"), model
    # fallback: local/other
    return OpenAI(api_key="EMPTY", base_url="backend"), model


# -----------------------------
# Paths per setting
# -----------------------------
def build_paths(setting: int, map_id: str, task_id: int, suffix: str | None) -> Tuple[str, str, str, str, str]:
    """Return map_path, task_path, world_json, agent_json, road_json."""
    # All settings now use the same structure as setting 1
    base_dir = os.path.join("multi_agent_world", "20_landmarks")
    map_path = os.path.join(base_dir, "maps")
    task_dir = os.path.join(base_dir, "tasks", f"map_road_20_{map_id}")
    default_suffix = "1"

    sfx = suffix if suffix is not None else default_suffix
    task_path = os.path.join(task_dir, f"task_dist_{task_id}_{sfx}")

    world_json = os.path.join(task_path, "progen_world.json")
    agent_json = os.path.join(task_path, "task_config.json")
    road_json  = os.path.join(task_path, "roads.json")
    return map_path, task_path, world_json, agent_json, road_json


# -----------------------------
# Core episode runners
# -----------------------------
def run_setting_1(env, info: Dict, observation: Dict, agent_1, agent_2, client, model,
                  map_id: str, task_id: int):
    # Agent 2 describes its view -> tournament match -> agent 1 rule-based to predicted location
    rgb2 = observation['rgb'][1]
    desc_text = describe_image(rgb2, client, model=model)

    dataset = load_dataset_from_info(info, with_pos=False)
    winner_id, _winner_img = tournament_winner(desc_text, dataset, client=client, model=model, shuffle=True)

    x, y = get_pos(winner_id, info)
    predict_location = [x, y]
    x2, y2, _ = tuple(info["agent"]["agent_location"][0])
    agent1_location = [x2, y2]

    if np.linalg.norm(np.array(agent1_location) - np.array(predict_location)) > 0.5:
        actions = agent_1.rule_based_navigation_actions(
            agent1_location, predict_location,
            info['agent']['agent_orientation'][0][1]
        )
        actions = actions[:-1]
        agent_1.update_ground_truth_action(actions)


def run_setting_2(env, info: Dict, observation: Dict, agent_1, agent_2, client, model,
                  map_id: str, task_id: int):
    # Agent 2 describes -> tournament picks target -> plan mid-node rendezvous
    rgb2 = observation['rgb'][1]
    desc_text = describe_image(rgb2, client, model=model)

    dataset = load_dataset_from_info(info, with_pos=False)
    winner_id, _ = tournament_winner(desc_text, dataset, client=client, model=model, shuffle=True)

    x, y = get_pos(winner_id, info)
    predict_location = [x, y]
    x2, y2, _ = tuple(info["agent"]["agent_location"][0])
    agent1_location = [x2, y2]

    _instr, _eval, nodes = agent_1.generate_instructions(
        predict_location, agent1_location,
        os.path.join(os.getcwd(), 'output', 'multi_agent', f'map_roud_{map_id}_{task_id}'), True
    )
    mid_node = nodes[len(nodes)//2]
    mid_location = [float(mid_node["pos"][0] * 100), float(mid_node["pos"][1] * 100)]
    closest_building = get_closest_building_by_midpoint(mid_location, info)

    # Agent 2 gets instructions to this building
    if np.linalg.norm(np.array(predict_location) - np.array(closest_building["building_position"])) > 0.5:
        agent2_instr, _se, _nds = agent_1.generate_instructions(
            predict_location, closest_building["building_position"],
            os.path.join(os.getcwd(), 'output', 'multi_agent', f'map_roud_{map_id}_{task_id}'), True
        )
        agent_2.set_instruction(agent2_instr)

    # Agent 1 rule-based to the same building
    if np.linalg.norm(np.array(agent1_location) - np.array(closest_building["building_position"])) > 0.5:
        actions = agent_1.rule_based_navigation_actions(
            agent1_location, closest_building["building_position"],
            info['agent']['agent_orientation'][0][1]
        )
        actions = actions[:-1]
        agent_1.update_ground_truth_action(actions)


def run_setting_3(env, info: Dict, observation: Dict, agent_1, agent_2, client, model,
                  map_id: str, task_id: int, fixed_target_id: int):
    # Similar to 2, but uses a fixed target building id
    rgb2 = observation['rgb'][1]
    _ = describe_image(rgb2, client, model=model)

    x, y = get_pos(fixed_target_id, info)
    tgt = [x, y]
    a1x, a1y, _ = info["agent"]["agent_location"][0]
    a1_loc = [a1x, a1y]

    _instr, _eval, nodes = agent_1.generate_instructions(
        tgt, a1_loc, os.path.join(os.getcwd(), 'output', 'multi_agent', f'map_roud_{map_id}_{task_id}'), True
    )
    mid = nodes[len(nodes)//2]
    mid_loc = [mid["pos"][0] * 100, mid["pos"][1] * 100]
    closest_b = get_closest_building_by_midpoint(mid_loc, info)
    closest   = get_closest_building_by_location(tgt, closest_b['building_position'], info)
    if closest["building_position"] == tgt:
        closest["building_position"] = a1_loc

    agent2_instr, _, _ = agent_1.generate_instructions(
        tgt, closest["building_position"],
        os.path.join(os.getcwd(), 'output', 'multi_agent', f'map_roud_{map_id}_{task_id}'), True
    )
    agent_2.set_instruction(agent2_instr)

    act_seq = agent_1.rule_based_navigation_actions(
        a1_loc, closest["building_position"], info['agent']['agent_orientation'][0][1]
    )
    agent_1.update_ground_truth_action(act_seq)


def main():
    setting = args.setting
    map_id = args.map
    task_id = args.task
    suffix = args.suffix

    # Resolve defaults for suffix if not provided
    if suffix is None:
        suffix = {1: "1", 2: "0_1", 3: "0_1"}[setting]

    # Env
    print(f"[INFO] Setting:{setting}  Map:{map_id}  Task:{task_id}_{suffix}  Port:{args.port}  Episodes:{args.episodes}")
    env = gym.make(
        'gym_citynav/BufferWorld-v0',
        port=args.port,
        observation_type="all",
        record_video=False,
        log_dir=args.log_dir,
        resolution=(320, 240),
        debug=args.debug,
    )

    # Paths
    map_path, task_path, world_json, agent_json, road_json = build_paths(setting, map_id, task_id, suffix)

    # LLM
    client, model = build_llm_client(args.backend, args.model)

    # Episodes
    for ep in range(args.episodes):
        options = {"task_path": task_path, "agent_json": agent_json, "map_path": map_path}
        # Pass world_json in settings that originally used it
        if setting in (1, 2):
            options["world_json"] = world_json

        observation, info = env.reset(options=options)

        # Agents
        if setting == 1:
            agent_1 = AgentClass(
                env.action_space, env.action_config, env.action_buffer,
                env.multi_agent_controller.agent_controllers[0],
                env.num_agents, 0, info["landmark_images"], info["buildings_info"],
                backend=args.backend, model=model
            )
            agent_2 = AgentClass(
                env.action_space, env.action_config, env.action_buffer,
                env.multi_agent_controller.agent_controllers[1],
                env.num_agents, 1, info["landmark_images"], info["buildings_info"],
                backend=args.backend, model=model
            )
        else:
            agent_1 = AgentClass(
                env.action_space, env.action_config, env.action_buffer,
                env.multi_agent_controller.agent_controllers[0],
                env.num_agents, 0, info["landmark_images"], info["buildings_info"]
            )
            agent_2 = AgentClass(
                env.action_space, env.action_config, env.action_buffer,
                env.multi_agent_controller.agent_controllers[1],
                env.num_agents, 1, info["landmark_images"], info["buildings_info"]
            )

        agent_1.load_env(misc.get_settingpath(world_json), misc.get_settingpath(road_json))

        # Setting-specific pre-planning
        if setting == 1:
            run_setting_1(env, info, observation, agent_1, agent_2, client, model, map_id, task_id)
        elif setting == 2:
            run_setting_2(env, info, observation, agent_1, agent_2, client, model, map_id, task_id)
        else:
            run_setting_3(env, info, observation, agent_1, agent_2, client, model, map_id, task_id, args.fixed_target_id)

        # Start agent threads
        agents = [agent_1, agent_2]
        agent_threads = []
        for i in range(env.num_agents):
            th = threading.Thread(target=agents[i].act, args=(False,))
            agent_threads.append(th)
            th.start()

        # Step loop
        terminated = False
        steps = 0
        while not terminated:
            observation, _, terminated, _, _info = env.step(None)
            time.sleep(0.05)
            steps += 1
            if args.max_steps and args.max_steps > 0 and steps >= args.max_steps:
                break

        # Stop agents
        for a in agents:
            a.is_finished = True
        for th in agent_threads:
            th.join()

        # Summary (use last _info if available)
        try:
            print(f"Episode:{ep} Steps:{_info['total_step']} Success:{_info['success']} Agent_Pos:{_info['agent']} Target_Pos:{_info['target']}")
        except Exception:
            print(f"Episode:{ep} Steps:{steps}")

    env.close()


if __name__ == "__main__":
    main()
