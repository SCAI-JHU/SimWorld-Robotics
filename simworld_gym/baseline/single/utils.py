import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import json
import cv2
import os

def numpy_to_base64(arr: np.ndarray) -> str:
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.dtype != np.uint8:
        arr = arr - np.min(arr)
        arr = arr / (np.max(arr) + 1e-8)
        arr = (arr * 255).astype(np.uint8)

    if len(arr.shape) == 2:
        img = Image.fromarray(arr, mode='L')
    elif arr.shape[2] == 1:
        img = Image.fromarray(arr[:, :, 0], mode='L')
    else:
        img = Image.fromarray(arr)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_str


def split_into_strips(arr: np.ndarray) -> list[str]:
    h, w = arr.shape[:2]
    third = w // 3
    strip1 = arr[:, :third]
    strip2 = arr[:, third:2*third]
    strip3 = arr[:, 2*third:]
    base64_strips = [
        numpy_to_base64(strip1),
        numpy_to_base64(strip2),
        numpy_to_base64(strip3)
    ]
    return base64_strips

def extract_action_dict(gpt_output: str):
    try:
        json_str = gpt_output.split("```json")[-1].strip("```")
    except:
        return False
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            cleaned = json_str.encode('utf-8').decode('unicode_escape')
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return False

action_mapping = ["Move_forward", "Rotate_left", "Rotate_right", "Move_left", "Move_right"]


def action_history_text(action_history, action_mapping):
    if len(action_history) == 0:
        return ""
    action_history_text = ""
    current_action = None
    n = 0
    for action in action_history:
        if current_action != action and current_action is not None:
            action_history_text += f"{action_mapping[current_action]} for {n} times,\n" if n > 1 else f"{action_mapping[current_action]},\n"
            current_action = action
            n = 1
        else:
            current_action = action
            n += 1
    action_history_text += f"{action_mapping[current_action]} for {n} times\n" if n > 1 else f"{action_mapping[current_action]}\n"
    return action_history_text

def display_images(current_view, expected_view):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    current_rgb = cv2.cvtColor(current_view, cv2.COLOR_BGR2RGB)
    expected_rgb = cv2.cvtColor(expected_view, cv2.COLOR_BGR2RGB)

    axes[0].imshow(current_rgb)
    axes[0].axis('off')
    axes[0].set_title('Current View')

    axes[1].imshow(expected_rgb)
    axes[1].axis('off')
    axes[1].set_title('Expected View')

    plt.tight_layout()
    plt.show()
    
def save_images(current_rgb: np.ndarray, expected_rgb: np.ndarray, save_path: str):
    """Save a side-by-side image for quick visual inspection."""
    try:
        h1, w1 = current_rgb.shape[:2]
        h2, w2 = expected_rgb.shape[:2]
        h = max(h1, h2)
        if h1 != h:
            scale = h / h1
            current_rgb = cv2.resize(current_rgb, (int(w1 * scale), h))
        if h2 != h:
            scale = h / h2
            expected_rgb = cv2.resize(expected_rgb, (int(w2 * scale), h))
        combo = cv2.hconcat([current_rgb, expected_rgb])
        cv2.imwrite(save_path, combo)
    except Exception:
        pass

def log_lines(folder_path: str, lines: list[tuple[str, str]]):
    with open(os.path.join(folder_path, "LLM_Output.txt"), "a") as f:
        for tag, content in lines:
            f.write(f"[{tag}]" + str(content) + "\n")