import base64
import cv2
import io
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

def safe_json_from_llm(text: str) -> Optional[Dict]:
    """
    Extract the first JSON object `{...}` from an LLM response and parse it.
    Supports responses wrapped in ```json ... ``` or containing extra text.
    Returns None on failure.
    """
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip("`")
    match = re.search(r"\{[\s\S]*?\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def extract_img_from_obs(obs):
    """Extract RGB image from observation dictionary."""
    if obs is None or not isinstance(obs, dict):
        return None

    vision = obs.get("vision")
    if isinstance(vision, dict):
        view = vision.get("egocentric view")
        if isinstance(view, dict):
            img = view.get("rgb")
            if isinstance(img, np.ndarray):
                return img
    return None


def save_img(img: np.ndarray, save_dir: str, frame_idx: int, agent_idx: int = None) -> bool:
    """Save image to disk with timestamp. Returns success status."""
    if img is None or not isinstance(img, np.ndarray):
        return False

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname = f"{frame_idx:06d}_{timestamp}.png"
    path = os.path.join(save_dir, fname)

    if img.shape[-1] == 3 and img.dtype == np.uint8:
        img_bgr = img[..., ::-1]
    else:
        img_bgr = img

    success = cv2.imwrite(path, img_bgr)
    if success and agent_idx is not None:
        print(f"[Agent-{agent_idx}] saved: {path}")
    return success


def extract_agent_obs(obs_all, agent_idx: int):
    """Extract observation for a specific agent from multi-agent observation."""
    if isinstance(obs_all, (list, tuple)):
        return obs_all[agent_idx]

    if isinstance(obs_all, dict) and "rgb" in obs_all:
        rgb_block = obs_all["rgb"]
        if isinstance(rgb_block, (list, tuple)):
            return {"rgb": rgb_block[agent_idx]}
        if agent_idx == 0:
            return {"rgb": rgb_block}
        
    for k in (agent_idx, str(agent_idx), f"Agent_{agent_idx}"):
        if isinstance(obs_all, dict) and k in obs_all:
            return obs_all[k]

    print("extract_agent_obs(): Unknown obs structure →", type(obs_all), obs_all.keys() if isinstance(obs_all, dict) else "")
    raise KeyError(f"Cannot find observation for agent {agent_idx}")


def _pil_to_b64(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def _img_to_b64(img: Union[str, np.ndarray]) -> str:
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.open(img).convert("RGB")
    return _pil_to_b64(im)

CAPTION_SYS = (
    "You are an expert building-description assistant.\n"
    "In ≤200 words, detail describe the building so another person could match photos of it.\n"
    "**Start the first sentence with the facade's MAIN COLOR and HEIGHT**.\n"
    "Cover these attributes as comma-separated phrases (each bullet includes an explanation):\n"
    "• main color — the dominant facade color\n"
    "• height — low / medium / tall (overall number of storeys)\n"
    "• primary materials — e.g. brick, concrete, glass, steel\n"
    "• window grid / pattern — shape & arrangement of windows\n"
    "• ground-floor storefront layout — doors, arches, glazing style\n"
    "• signage text — exact words visible; say 'no signage' if none\n"
    "• sidewalk objects — lamp-post, tree, bench, trash can, etc.\n"
    "• distinctive features — anything that makes this building unique\n"
    "• neighbor — anything around it\n"
    "Keep it factual, detail and avoid subjective opinions. Keep a natural language form. "
)

def describe_image(img: Union[str, np.ndarray], client, model) -> str:
    b64 = _img_to_b64(img)
    messages = [
        {"role": "system", "content": CAPTION_SYS},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]},
    ]
    resp = client.chat.completions.create(model=model, temperature=0.1, max_tokens=256, messages=messages)
    caption = resp.choices[0].message.content.strip()

    os.makedirs("debug_caption", exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    if isinstance(img, np.ndarray):
        import cv2
        fname_img = f"debug_caption/img_{timestamp}.png"
        cv2.imwrite(fname_img, img)

    fname_txt = f"debug_caption/desc_{timestamp}.txt"
    with open(fname_txt, "w", encoding="utf-8") as f:
        f.write(caption)

    return caption

_MATCH_SYS = (
    "TEXT = natural-language description of the target building.\n"
    "Two FULL-FACADE candidate photos are shown: A (first) and B (second).\n"
    "Decide which matches TEXT better.\n"
    "Give STRONG weight to facade color, signs, materials and distinctive features."
    "If facade color clearly mismatches, that candidate must lose.\n"
    "Reply ONLY 'A' or 'B'."
)

def tie_break(desc_text: str,
                    a: Tuple[int, str],
                    b: Tuple[int, str], client, model) -> int:
    messages = [
        {"role": "system", "content": _MATCH_SYS},
        {"role": "user", "content": [
            {"type": "text", "text": desc_text},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_img_to_b64(a[1])}"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{_img_to_b64(b[1])}"}},
        ]},
    ]
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=1, temperature=0)
    choice = resp.choices[0].message.content.strip().upper()
    return a[0] if "A" in choice else b[0]

def tournament_winner(desc_text: str,
                            candidates: List[Tuple[int, str]], client, model,
                            shuffle: bool = True) -> Tuple[int, str]:
    """
    Knock‑out tournament until one (id, path) remains.
    """
    if shuffle:
        random.shuffle(candidates)

    while len(candidates) > 1:
        next_round: List[Tuple[int, str]] = []
        i = 0
        while i < len(candidates):
            a = candidates[i]
            if i == len(candidates) - 1:        # bye
                next_round.append(a)
            else:
                b = candidates[i + 1]
                winner_id = tie_break(desc_text, a, b, client, model)
                winner = a if winner_id == a[0] else b
                next_round.append(winner)
            i += 2
        candidates = next_round
    return candidates[0]   # (id, numpy.ndarray)


def load_dataset_from_info(
    info: Dict,
    with_pos: bool = True,
) -> List[Tuple[int, np.ndarray] | Tuple[int, np.ndarray, Tuple[float, float]]]:
    imgs       = info["landmark_images"]      # List[np.ndarray]
    meta       = info["buildings_info"]       # List[Dict]
    assert len(imgs) == len(meta), "unmatched images and metadata"

    dataset = []
    for img, m in zip(imgs, meta):
        bid  = int(m["building_id"])
        if with_pos:
            x, y = map(float, m["building_position"][:2])
            dataset.append((bid, img, (x, y)))
        else:
            dataset.append((bid, img))
    
    dataset.sort(key=lambda x: x[0])
    return dataset

def get_pos(bid: int, info: dict) -> Tuple[float, float]:
    if "_id2pos" not in info:
        id2pos = {}
        for m in info["buildings_info"]:
            id2pos[int(m["building_id"])] = tuple(map(float, m["building_position"][:2]))
        info["_id2pos"] = id2pos

    try:
        return info["_id2pos"][bid]
    except KeyError:
        raise KeyError(f"building_id={bid} not found in info['buildings_info']")

def get_closest_building_by_location(
    loc1: List[float] | Tuple[float, float],
    loc2: List[float] | Tuple[float, float],
    info: Dict
) -> Dict:
    mid = [(loc1[0] + loc2[0]) / 2, (loc1[1] + loc2[1]) / 2]
    mid_xy = np.array(mid)

    best = None
    best_d = float("inf")

    for b in info["buildings_info"]:
        b_xy = np.array(b["building_position"][:2])
        d = np.linalg.norm(b_xy - mid_xy)
        if d < best_d:
            best_d = d
            best = b

    return best

def get_closest_building_by_midpoint(
    mid_location: List[float] | Tuple[float, float],
    info: Dict
) -> Dict:
    meet_xy = np.array(mid_location)

    best = None
    best_d = float("inf")

    for b in info["buildings_info"]:
        b_xy = np.array(b["building_position"][:2])
        d = np.linalg.norm(b_xy - meet_xy)
        if d < best_d:
            best_d = d
            best = b

    return best
