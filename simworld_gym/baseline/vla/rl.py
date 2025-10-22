#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import random
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from baseline.single.utils import *

from transformers import AutoTokenizer

import gym
import gym_citynav

import subprocess, socket
from contextlib import closing

# -----------------------------
# Config
# -----------------------------
LANG_MODEL = "microsoft/deberta-v3-base"
IMG_MODEL  = "facebook/dinov2-base"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

NUM_ACTIONS = 4
ENV_ACTION = {0: -1, 1: 0, 2: 5, 3: 4}
IDX2TEXT   = {0: "Subtask complete", 1: "Move forward", 2: "Turn left", 3: "Turn right"}

NUM_ENVS   = int(os.getenv("NUM_ENVS", "2"))       # 1 or 2
BASE_PORT  = int(os.getenv("UE_PORT", "9400"))
RESOLUTION = (720, 600)

ROLLOUT_STEPS_MAX = int(os.getenv("ROLLOUT_STEPS", "512"))
ROLLOUT_STEPS_MIN = 128
ROLLOUT_STEPS_INIT = ROLLOUT_STEPS_MIN

EPOCHS        = int(os.getenv("PPO_EPOCHS", "10"))
MINIBATCH     = int(os.getenv("MINIBATCH", "32"))
GAMMA         = 0.99
LAMBDA        = 0.95
CLIP_EPS      = 0.2
ENT_COEF      = 0.01
VF_COEF       = 0.5
LR            = 3e-4
MAX_GRAD_NORM = 0.5
MAX_TEXT_LEN  = 256

SAVE_DIR      = os.getenv("SAVE_DIR", "ppo_checkpoints")
IL_INIT_CKPT  = os.getenv("IL_INIT", "best_policy_model.pt")

# eval & stability
EVAL_INTERVAL   = int(os.getenv("EVAL_INTERVAL", "5"))   # evaluate every 5 updates
NUM_EVAL_WORLDS = int(os.getenv("NUM_EVAL_WORLDS", "10"))
EPISODES_PER_WORLD = int(os.getenv("EPISODES_PER_WORLD", "2"))
MIN_EP_SECONDS  = float(os.getenv("MIN_EP_SECONDS", "5.0"))  # per-task min wall time

torch.backends.cudnn.benchmark = True

# -----------------------------
# Policy (actor + critic)
# -----------------------------
from baseline.vla.imitation import MultimodalPolicy  # your multimodal attention policy

class ActorCritic(nn.Module):
    def __init__(self, lang_model=LANG_MODEL, img_model=IMG_MODEL,
                 d_model=768, nheads=8, p_drop=0.1, p_tokdrop=0.1):
        super().__init__()
        self.core = MultimodalPolicy(lang_model=lang_model, img_model=img_model,
                                     d_model=d_model, nheads=nheads,
                                     p_drop=p_drop, p_tokdrop=p_tokdrop)
        self.value_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, input_ids, attention_mask, cur_img, exp_img):
        logits, pooled = self.forward_with_features(input_ids, attention_mask, cur_img, exp_img)
        value = self.value_head(pooled).squeeze(-1)
        return logits, value

    def forward_with_features(self, input_ids, attention_mask, cur_img, exp_img):
        # text
        txt_last = self.core.txt(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        txt_tok  = self.core.drop_proj(self.core.proj_txt(txt_last))
        # vision
        cur_px = self.core.img_proc(images=cur_img, return_tensors="pt")["pixel_values"].to(txt_tok.device)
        exp_px = self.core.img_proc(images=exp_img, return_tensors="pt")["pixel_values"].to(txt_tok.device)
        vcur_tok = self.core.drop_proj(self.core.proj_vis(self.core.vcur(pixel_values=cur_px).last_hidden_state))
        vexp_tok = self.core.drop_proj(self.core.proj_vis(self.core.vexp(pixel_values=exp_px).last_hidden_state))
        # concat & mask
        seq = torch.cat([vcur_tok, vexp_tok, txt_tok], dim=1)
        B, Tc, Te, Tt = vcur_tok.size(0), vcur_tok.size(1), vexp_tok.size(1), txt_tok.size(1)
        mask_valid = torch.cat([
            torch.ones(B, Tc, device=seq.device, dtype=attention_mask.dtype),
            torch.ones(B, Te, device=seq.device, dtype=attention_mask.dtype),
            attention_mask
        ], dim=1)
        if self.training and self.core.p_tokdrop > 0.0:
            seq, mask_valid = self.core.token_dropout(seq, mask_valid, self.core.p_tokdrop)
        key_pad = ~mask_valid.bool()
        attn_out, _ = self.core.attn(seq, seq, seq, key_padding_mask=key_pad)
        pooled = self.core.masked_mean(attn_out, mask_valid)
        logits = self.core.head(pooled)
        return logits, pooled

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL, use_fast=False)

# -----------------------------
# World split & alternating tasks
# -----------------------------
def make_world_name(i: int) -> str:
    return f"map_road_20_{i}"

TASKS_PAIR = ["task_dist_11_0_1", "task_dist_21_0_1"]

def build_world_splits():
    train_idx = list(range(0, 25)) + list(range(75, 100))
    test_idx  = list(range(25, 75))
    train_worlds = [make_world_name(i) for i in train_idx]
    test_worlds  = [make_world_name(i) for i in test_idx]
    return train_worlds, test_worlds

class WorldSampler:
    def __init__(self, worlds: List[str], shuffle=True):
        self.worlds = worlds[:]
        self.shuffle = shuffle
        self.i = 0
        if self.shuffle:
            random.shuffle(self.worlds)

    def next_world(self) -> str:
        if self.i >= len(self.worlds):
            self.i = 0
            if self.shuffle:
                random.shuffle(self.worlds)
        w = self.worlds[self.i]
        self.i += 1
        return w

class WorldCycler:
    """Keep a world for 'episodes_per_world' episodes, alternate the two tasks.
    Only reload world_json when switching to a new world."""
    def __init__(self, sampler: WorldSampler, episodes_per_world: int = 10):
        self.sampler = sampler
        self.episodes_per_world = episodes_per_world
        self.env_state = {}  # env_id -> dict(world, task_idx, epi_cnt)

    def init_env(self, env_id: int):
        w = self.sampler.next_world()
        self.env_state[env_id] = {"world": w, "task_idx": 0, "epi_cnt": 0}

    def first_reset_args(self, env_id: int):
        st = self.env_state[env_id]
        return st["world"], TASKS_PAIR[st["task_idx"]], True  # first time -> reload world

    def next_reset(self, env_id: int):
        st = self.env_state[env_id]
        world_changed = False
        if st["epi_cnt"] >= self.episodes_per_world:
            st["world"] = self.sampler.next_world()
            st["task_idx"] = 0
            st["epi_cnt"] = 0
            world_changed = True
        else:
            st["task_idx"] ^= 1  # alternate task
        st["epi_cnt"] += 1
        return st["world"], TASKS_PAIR[st["task_idx"]], world_changed

    def current(self, env_id: int):
        st = self.env_state[env_id]
        return st["world"], TASKS_PAIR[st["task_idx"]]

# -----------------------------
# Env helpers
# -----------------------------
def make_env(port: int, log_dir: str):
    reward_setting = {
        "human_collision_penalty": -1.0,
        "object_collision_penalty": -0.1,
        "action_penalty": -0.1,
        "success_reward": 100.0,
        "subtask_penalty": -10.0,
        "subtask_reward": 10.0,
        "off_track_penalty": -0.002,
    }
    env = gym.make(
        "gym_citynav/SimpleWorld-v0",
        verbose=False,
        log_level="ERROR",            # quiet
        port=port,
        resolution=RESOLUTION,
        render_mode="rgb_array",
        observation_type="all",
        record_video=False,
        log_dir=log_dir,
        reward_setting=reward_setting,
    )
    return env

def to_pil_rgb(x):
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[:, :, ::-1]  # BGR->RGB
    return Image.fromarray(arr)

def build_text(instruction: str, history_list: List[int], orientation) -> str:
    return f"Instruction: {instruction} HISTORY: {action_history_text(history_list, IDX2TEXT)} ORIENTATION: {orientation}"

# -----------------------------
# Rollout buffer
# -----------------------------
@dataclass
class StepSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    cur_img: Image.Image
    exp_img: Image.Image
    action: int
    logp: float
    value: float
    reward: float
    done: bool

class RolloutBuffer:
    def __init__(self, T, N):
        self.T = T
        self.N = N
        self.storage: List[List[StepSample]] = [[] for _ in range(N)]

    def add(self, env_id: int, sample: StepSample):
        self.storage[env_id].append(sample)

    def build_tensors(self, device):
        batch = []
        for n in range(self.N):
            batch.extend(self.storage[n])
        input_ids = torch.stack([s.input_ids.squeeze(0) for s in batch]).to(device)
        attn      = torch.stack([s.attention_mask.squeeze(0) for s in batch]).to(device)
        actions   = torch.tensor([s.action for s in batch], device=device)
        logp_old  = torch.tensor([s.logp for s in batch], dtype=torch.float32, device=device)
        values    = torch.tensor([s.value for s in batch], dtype=torch.float32, device=device)
        rewards   = torch.tensor([s.reward for s in batch], dtype=torch.float32, device=device)
        dones     = torch.tensor([s.done for s in batch], dtype=torch.float32, device=device)
        cur_imgs  = [s.cur_img for s in batch]
        exp_imgs  = [s.exp_img for s in batch]
        return {
            "input_ids": input_ids,
            "attn": attn,
            "actions": actions,
            "logp_old": logp_old,
            "values": values,
            "rewards": rewards,
            "dones": dones,
            "cur_imgs": cur_imgs,
            "exp_imgs": exp_imgs,
        }

# -----------------------------
# PPO utils
# -----------------------------
def compute_gae(rollout: dict, bootstrap_value: torch.Tensor, gamma=GAMMA, lam=LAMBDA, T=64, N=1):
    rewards = rollout["rewards"].view(T, N)
    dones   = rollout["dones"].view(T, N)
    values  = rollout["values"].view(T, N)
    adv = torch.zeros_like(rewards, device=rewards.device)
    lastgaelam = torch.zeros(N, device=rewards.device)
    for t in reversed(range(T)):
        nextnonterm = 1.0 - (dones[t])
        nextvalue = bootstrap_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalue * nextnonterm - values[t]
        lastgaelam = delta + gamma * lam * nextnonterm * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return adv.flatten(), returns.flatten()

def ppo_update(model: ActorCritic, optimizer, rollout: dict, advantages, returns,
               clip_eps=CLIP_EPS, vf_coef=VF_COEF, ent_coef=ENT_COEF,
               epochs=EPOCHS, minibatch_size=MINIBATCH, max_grad_norm=MAX_GRAD_NORM):
    B = rollout["actions"].shape[0]
    idxs = np.arange(B)
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for start in range(0, B, minibatch_size):
            mb_idx = idxs[start:start+minibatch_size]
            ids   = rollout["input_ids"][mb_idx]
            attn  = rollout["attn"][mb_idx]
            acts  = rollout["actions"][mb_idx]
            logp0 = rollout["logp_old"][mb_idx]
            rets  = returns[mb_idx]
            advs  = advantages[mb_idx]
            cur_list = [rollout["cur_imgs"][i] for i in mb_idx]
            exp_list = [rollout["exp_imgs"][i] for i in mb_idx]
            logits, values = model(ids, attn, cur_list, exp_list)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(acts)
            entropy = dist.entropy().mean()
            ratio = (logp - logp0).exp()
            pg1 = ratio * advs
            pg2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advs
            policy_loss = -torch.min(pg1, pg2).mean()
            value_loss = 0.5 * (rets - values).pow(2).mean()
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

# -----------------------------
# Adaptive rollout stats
# -----------------------------
class Stats:
    def __init__(self, cap=200):
        self.ep_lens = []
        self.cap = cap
    def add_ep_len(self, l):
        self.ep_lens.append(l)
        if len(self.ep_lens) > self.cap:
            self.ep_lens = self.ep_lens[-self.cap:]
    def mean_ep_len(self):
        if not self.ep_lens:
            return None
        return int(np.mean(self.ep_lens))

def adapt_rollout_length(current_T, stats: Stats):
    m = stats.mean_ep_len()
    if m is None:
        return current_T
    target = int(min(ROLLOUT_STEPS_MAX, max(ROLLOUT_STEPS_MIN, 20 * m // 2)))
    new_T = int(0.5 * current_T + 0.5 * target)
    new_T = max(ROLLOUT_STEPS_MIN, min(ROLLOUT_STEPS_MAX, (new_T // 64) * 64))
    return new_T

# -----------------------------
# Reset helpers (stamp start time)
# -----------------------------
def reset_env(env, world: str, task: str, world_changed: bool):
    options = {
        "task_path": os.path.join("single_agent_world", "easy", world, task),
        "agent_json": os.path.join("single_agent_world", "easy", world, task, "task_config.json"),
    }
    if world_changed:
        options["world_json"] = os.path.join("single_agent_world", "easy", world, task, "progen_world.json")

    obs, info = env.reset(options=options)
    state = {
        "history": [],
        "exp_img": info["current_instruction"]["image"],
        "instruction": info["current_instruction"]["text"],
        "orientation": str(info["agent"]["agent_rotation"]),
        "world": world,
        "task": task,
        "ep_len": 0,
        "ep_start": time.time(),   # wall-clock start
    }
    return obs, info, state

def safe_reset_call(env_id, envs, world, task, world_changed, ue_supervisor, cycler):
    try:
        return reset_env(envs[env_id], world, task, world_changed)
    except RECOVERABLE_EXC as e:
        print(f"[warn] env {env_id} reset failed: {e!r}. Recovering...")
        return recover_env(env_id, envs, ue_supervisor, cycler, keep_same_world_task=True)

# -----------------------------
# UE Restarter
# -----------------------------
class UERestarter:
    def __init__(self, entries, ue_bin="/Linux/gym_citynav.sh", cwd="/Linux"):
        self.entries = entries
        self.ue_bin = ue_bin
        self.cwd = cwd
        self.procs = [None] * len(entries)

    @staticmethod
    def _tcp_ping(port, host="127.0.0.1", timeout=0.5):
        try:
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(timeout)
                s.connect((host, port))
            return True
        except Exception:
            return False

    def start(self, env_id):
        cfg = self.entries[env_id]
        env = os.environ.copy()
        env["DISPLAY"] = cfg["display"]
        env["UE_PORT"] = str(cfg["port"])
        log_f = open(cfg["log"], "ab", buffering=0)
        self.procs[env_id] = subprocess.Popen(
            [self.ue_bin, "-UnrealCVPort", str(cfg["port"]), "-RenderOffScreen"],
            cwd=self.cwd,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )

    def ensure_running(self, env_id, wait_ready_seconds=45, poll_every=0.5):
        need_restart = False
        p = self.procs[env_id]
        if p is None or (p.poll() is not None):
            need_restart = True
        if need_restart:
            self.start(env_id)
        deadline = time.time() + wait_ready_seconds
        port = self.entries[env_id]["port"]
        while time.time() < deadline:
            if self._tcp_ping(port):
                return True
            time.sleep(poll_every)
        return False

    def kill(self, env_id):
        p = self.procs[env_id]
        if p is not None:
            try:
                p.terminate(); p.wait(timeout=5)
            except Exception:
                try: p.kill()
                except Exception: pass
            self.procs[env_id] = None

def safe_recreate_env(old_env, port: int, log_dir: str):
    try:
        old_env.close()
    except Exception:
        pass
    return make_env(port, log_dir=log_dir)

RECOVERABLE_EXC = (ConnectionError, ConnectionResetError, RuntimeError, OSError, Exception)

def recover_env(env_id: int,
                envs: list,
                ue_supervisor: UERestarter,
                cycler: WorldCycler,
                keep_same_world_task: bool = True):
    """Close env, ensure UE is running, recreate env, and reset to desired world/task."""
    print(f"[recover] env {env_id}: closing env and restarting UE...")
    # 1) close
    try:
        envs[env_id].close()
    except Exception:
        pass

    # 2) (re)start UE process and wait until port is ready
    ok = ue_supervisor.ensure_running(env_id, wait_ready_seconds=60)
    if not ok:
        raise RuntimeError(f"UE for env {env_id} did not become ready after restart")

    # 3) recreate env bound to the same port
    port = ue_supervisor.entries[env_id]["port"]
    envs[env_id] = safe_recreate_env(envs[env_id], port=port, log_dir=f"ppo_env_{env_id}")

    # 4) reset: either same world/task (fast) or the next scheduled one
    if keep_same_world_task:
        w_cur, t_cur = cycler.current(env_id)
        obs, info, st = reset_env(envs[env_id], w_cur, t_cur, world_changed=False)
    else:
        w_next, t_next, world_changed = cycler.next_reset(env_id)
        obs, info, st = reset_env(envs[env_id], w_next, t_next, world_changed)

    # help downstream (wall-time guards etc.)
    st["ep_start"] = time.time()
    return obs, info, st

# -----------------------------
# Collector (alternate tasks + per-task min wall time)
# -----------------------------
def collector_rollout(model: ActorCritic,
                      envs: List[gym.Env],
                      cycler: WorldCycler,
                      tokenizer: AutoTokenizer,
                      T: int,
                      device=DEVICE,
                      stats: Stats = None,
                      ue_supervisor: UERestarter = None):

    assert ue_supervisor is not None, "Pass ue_supervisor to enable auto-recovery"

    N = len(envs)
    for i in range(N):
        cycler.init_env(i)

    # initial reset per env
    obs_list, info_list, states = [], [], []
    for i in range(N):
        w, t, world_changed = cycler.first_reset_args(i)
        try:
            obs, info, st = reset_env(envs[i], w, t, world_changed)
        except RECOVERABLE_EXC as e:
            print(f"[warn] env {i} initial reset failed: {e!r}. Recovering...")
            obs, info, st = recover_env(i, envs, ue_supervisor, cycler, keep_same_world_task=True)
        st["ep_start"] = time.time()
        obs_list.append(obs); info_list.append(info); states.append(st)

    buf = RolloutBuffer(T, N)

    for tstep in tqdm(range(T), desc="rollout"):
        # build batch
        ids_batch, attn_batch, cur_imgs, exp_imgs = [], [], [], []
        for i in range(N):
            txt = build_text(states[i]["instruction"], states[i]["history"], states[i]["orientation"])
            enc = tokenizer(txt, padding="max_length", truncation=True, max_length=MAX_TEXT_LEN, return_tensors="pt")
            ids_batch.append(enc["input_ids"]); attn_batch.append(enc["attention_mask"])
            cur_imgs.append(to_pil_rgb(obs_list[i]["rgb"]))
            exp_imgs.append(to_pil_rgb(states[i]["exp_img"]))
        input_ids = torch.cat(ids_batch, dim=0).to(device)
        attn_mask = torch.cat(attn_batch, dim=0).to(device)

        logits, values = model(input_ids, attn_mask, cur_imgs, exp_imgs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        logps   = dist.log_prob(actions)

        # step each env with protection
        next_obs_list, next_info_list = [], []
        for i in range(N):
            a = int(actions[i].item())
            env_act = ENV_ACTION[a]

            try:
                next_obs, rew, terminated, truncated, next_info = envs[i].step(env_act)
                step_ok = True
            except RECOVERABLE_EXC as e:
                print(f"[warn] env {i} step failed: {e!r}. Recovering...")
                # Recover the env (same world/task fast reset)
                next_obs, next_info, states[i] = recover_env(i, envs, ue_supervisor, cycler, keep_same_world_task=True)
                # Option A: drop this transition entirely (no buf.add)
                step_ok = False
                # Option B (alternative): add a neutral transition with 0 reward & done=False

            if step_ok:
                done = bool(terminated or truncated)
                buf.add(i, StepSample(
                    input_ids=input_ids[i:i+1].detach().cpu(),
                    attention_mask=attn_mask[i:i+1].detach().cpu(),
                    cur_img=cur_imgs[i],
                    exp_img=exp_imgs[i],
                    action=a,
                    logp=float(logps[i].item()),
                    value=float(values[i].item()),
                    reward=float(rew),
                    done=done,
                ))

                # bookkeeping only when the step succeeded
                states[i]["history"].append(a)
                states[i]["ep_len"] += 1

                # subtask advanced? refresh expected image/instruction
                if env_act == -1 and not done:
                    states[i]["exp_img"]     = next_info["current_instruction"]["image"]
                    states[i]["instruction"] = next_info["current_instruction"]["text"]
                    states[i]["orientation"] = str(next_info["agent"]["agent_rotation"])
                    states[i]["history"]     = []

                if done:
                    if stats is not None:
                        stats.add_ep_len(states[i]["ep_len"])
                    # alternate task within world or switch world per cycler
                    w, tname, world_changed = cycler.next_reset(i)
                    # To reduce wall-time, only reload world when it changes
                    try:
                        next_obs, next_info, states[i] = reset_env(envs[i], w, tname, world_changed)
                    except RECOVERABLE_EXC as e:
                        print(f"[warn] env {i} reset after done failed: {e!r}. Recovering...")
                        next_obs, next_info, states[i] = recover_env(i, envs, ue_supervisor, cycler, keep_same_world_task=True)

                    states[i]["ep_start"] = time.time()  # restart timer

            next_obs_list.append(next_obs)
            next_info_list.append(next_info)

        obs_list = next_obs_list
        info_list = next_info_list

    # bootstrap values
    ids_batch, attn_batch, cur_imgs, exp_imgs = [], [], [], []
    for i in range(N):
        txt = build_text(states[i]["instruction"], states[i]["history"], states[i]["orientation"])
        enc = tokenizer(txt, padding="max_length", truncation=True, max_length=MAX_TEXT_LEN, return_tensors="pt")
        ids_batch.append(enc["input_ids"]); attn_batch.append(enc["attention_mask"])
        cur_imgs.append(to_pil_rgb(obs_list[i]["rgb"]))
        exp_imgs.append(to_pil_rgb(states[i]["exp_img"]))
    input_ids = torch.cat(ids_batch, dim=0).to(device)
    attn_mask = torch.cat(attn_batch, dim=0).to(device)
    _, bootstrap_values = model(input_ids, attn_mask, cur_imgs, exp_imgs)

    rollout = buf.build_tensors(device)
    adv, rets = compute_gae(rollout, bootstrap_values.detach(), T=T, N=N)
    return rollout, adv, rets


# -----------------------------
# Evaluation (sample subset + minimal wall time + minimal world reload)
# -----------------------------
def evaluate(model: ActorCritic, env: gym.Env, test_worlds: List[str],
             sample_k: int = 10, print_details: bool = False, seed: int = None, ue_supervisor: UERestarter = None):
    model.eval()
    rng = random.Random(seed)
    chosen = test_worlds if sample_k <= 0 or sample_k >= len(test_worlds) else rng.sample(test_worlds, sample_k)

    success = 0
    ep_rewards = []

    for world in tqdm(chosen):
        for j, task in enumerate(TASKS_PAIR):
            # reload world only for the first task of a world
            obs, info, state = reset_env(env, world, task, world_changed=(j == 0))
                
            history = []
            exp_img = info["current_instruction"]["image"]
            instruction = info["current_instruction"]["text"]
            orientation = str(info["agent"]["agent_rotation"])

            done = False
            term_flag = False
            R = 0.0

            while not done:
                txt = build_text(instruction, history, orientation)
                enc = tokenizer(txt, padding="max_length", truncation=True, max_length=MAX_TEXT_LEN, return_tensors="pt")
                ids = enc["input_ids"].to(DEVICE); attn = enc["attention_mask"].to(DEVICE)
                cur_pil = to_pil_rgb(obs["rgb"]); exp_pil = to_pil_rgb(exp_img)

                logits, _ = model.forward_with_features(ids, attn, [cur_pil], [exp_pil])
                action = int(torch.argmax(logits, dim=-1).item())
                env_act = ENV_ACTION[action]

                obs, rew, term, trunc, info = env.step(env_act)
                R += float(rew)

                history.append(action)
                if env_act == -1 and not (term or trunc):
                    exp_img = info["current_instruction"]["image"]
                    instruction = info["current_instruction"]["text"]
                    orientation = str(info["agent"]["agent_rotation"])
                    history = []

                done = bool(term or trunc)
                term_flag = term

            # per-task min wall time on eval as well
            elapsed = time.time() - state["ep_start"]
            if elapsed < 10:
                time.sleep(10 - elapsed)

            ep_rewards.append(R)
            success += 1 if rew >= 50 else 0
            if print_details:
                print(f"[eval] world={world} task={task}  R={R:.2f}")

    denom = len(chosen) * len(TASKS_PAIR)
    sr = success / denom if denom > 0 else 0.0
    avg_R = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    if print_details:
        print(f"[eval] summary: episodes={denom}  SR={sr:.3f}  AvgR={avg_R:.2f}")
    return sr, avg_R

# -----------------------------
# Checkpoint helpers (crash-safe)
# -----------------------------
def ckpt_path(name: str) -> str:
    return os.path.join(SAVE_DIR, name)

def save_checkpoint(model, optimizer, update_idx, current_T, tag="latest"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = ckpt_path(f"{tag}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update_idx": update_idx,
        "current_T": current_T
    }, path)
    return path

def try_load_latest(model, optimizer):
    latest = ckpt_path("latest.pt")
    if os.path.isfile(latest):
        try:
            ckpt = torch.load(latest, map_location="cpu")
            model.load_state_dict(ckpt["model"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer"])
            update_idx = ckpt.get("update_idx", 0)
            current_T  = ckpt.get("current_T", ROLLOUT_STEPS_INIT)
            print(f"[resume] loaded {latest} (update={update_idx}, T={current_T})")
            return update_idx, current_T, True
        except Exception as e:
            print(f"[resume] failed to load {latest}: {e}")
    return 0, ROLLOUT_STEPS_INIT, False

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    train_worlds, test_worlds = build_world_splits()
    sampler = WorldSampler(train_worlds, shuffle=True)
    cycler  = WorldCycler(sampler, episodes_per_world=EPISODES_PER_WORLD)

    ports = [BASE_PORT + i for i in range(NUM_ENVS)]
    displays = [os.getenv("DISP1", ":120"), os.getenv("DISP2", ":121")]
    ue_entries = []
    print("[ports]", ports)
    
    for i in range(NUM_ENVS):
        ue_entries.append({
            "env_id": i,
            "port": ports[i],
            "display": displays[i if i < len(displays) else -1],
            "log": f"/SimWorld/unreal{i}.log",
        })
    ue_supervisor = UERestarter(ue_entries)
    for i in range(NUM_ENVS):
        ok = ue_supervisor.ensure_running(i, wait_ready_seconds=60)
        if not ok:
            raise RuntimeError(f"UE {i} on port {ue_entries[i]['port']} not ready")
    print("Connection OK!")
    envs = [make_env(ports[i], log_dir=f"ppo_env_{i}") for i in range(NUM_ENVS)]
    eval_env = envs[0]  # reuse one env for eval
    
    model = ActorCritic().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # IL warm start (optional)
    if IL_INIT_CKPT and os.path.isfile(IL_INIT_CKPT):
        try:
            sd = torch.load(IL_INIT_CKPT, map_location="cpu")
            missing, unexpected = model.core.load_state_dict(sd, strict=False)
            print("[IL warm-start] loaded:", IL_INIT_CKPT)
            if missing:   print("  missing:", missing)
            if unexpected:print("  unexpected:", unexpected)
        except Exception as e:
            print(f"[IL warm-start] failed to load {IL_INIT_CKPT}: {e}")

    # resume training if latest exists
    update_idx, current_T, resumed = try_load_latest(model, optimizer)

    stats = Stats()

    try:
        while True:
            model.train()
            rollout, advantages, returns = collector_rollout(
                model, envs, cycler, tokenizer, T=current_T, device=DEVICE, stats=stats, ue_supervisor=ue_supervisor
            )

            ppo_update(model, optimizer, rollout, advantages, returns,
                       clip_eps=CLIP_EPS, vf_coef=VF_COEF, ent_coef=ENT_COEF,
                       epochs=EPOCHS, minibatch_size=MINIBATCH, max_grad_norm=MAX_GRAD_NORM)

            update_idx += 1

            # Save after EVERY update (crash-safe)
            save_checkpoint(model, optimizer, update_idx, current_T, tag="latest")

            # Evaluate every EVAL_INTERVAL updates on a random subset of worlds
            if update_idx % EVAL_INTERVAL == 0:
                time.sleep(10)
                sr, avgR = evaluate(model, eval_env, test_worlds,
                                    sample_k=NUM_EVAL_WORLDS, print_details=False, seed=update_idx, ue_supervisor=ue_supervisor)
                ckpt = ckpt_path(f"ppo_update_{update_idx}_sr{sr:.3f}_R{avgR:.1f}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update_idx": update_idx,
                    "current_T": current_T
                }, ckpt)
                print(f"[eval] update={update_idx}  SR={sr:.3f}  AvgR={avgR:.2f}  saved={ckpt}")

            # adapt rollout horizon
            current_T = adapt_rollout_length(current_T, stats)

            # optional stop
            if update_idx >= 1000:
                print("[train] reached max updates; final save.")
                save_checkpoint(model, optimizer, update_idx, current_T, tag="final")
                return

    except KeyboardInterrupt:
        print("\n[train] KeyboardInterrupt – saving latest...")
        save_checkpoint(model, optimizer, update_idx, current_T, tag="latest")
        raise
    except Exception as e:
        print(f"\n[train] Exception caught: {e}\nSaving emergency checkpoint...")
        save_checkpoint(model, optimizer, update_idx, current_T, tag="emergency")
        raise
    finally:
        # ensure a final save on orderly exit
        save_checkpoint(model, optimizer, update_idx, current_T, tag="latest")

if __name__ == "__main__":
    main()
