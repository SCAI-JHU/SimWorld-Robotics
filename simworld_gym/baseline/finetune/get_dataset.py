import json
import os
from datasets import Dataset, Features, Value, ClassLabel, Image, DatasetDict
from baseline.single.utils import action_history_text
from huggingface_hub import upload_folder, create_repo
from glob import glob
import random

random.seed(42)

features = Features({
    "id": Value("string"),
    "subtask": Value("string"),
    "orientation": Value("string"),
    "history": Value("string"),
    "target_orientation": Value("string"),
    "distance": Value("string"),
    "current_view": Image(decode=True),
    "expected_view": Image(decode=True),
    "action": ClassLabel(names=["Subtask completed", "Move forward", "Turn left", "Turn right"]),
    "plan": Value("string")
})

all_steps = glob("../task_generator/output/single_world/train_dataset/easy_mode/map_road_*/task_dist_*/training_step_*/meta_data.json")

step_action = {-1: [], 0: [], 4: [], 5: []}

for step in all_steps:
    with open(step, "r") as f:
        sample = json.load(f)
    step_action[sample["ground_truth_action"][0]].append(step)


print([(action, len(step)) for action, step in step_action.items()])

steps = []
steps.extend(step_action[-1])
steps.extend(random.sample(step_action[0], 2000))
steps.extend(step_action[4])
steps.extend(step_action[5])

data = []
i = 0
action_mapping = {
    0: "Move forward",
    5: "Turn left",
    4: "Turn right",
    -1: "Subtask completed",
}

def reveal_position(orientation, action):
    if action == 5:
        if orientation == "East":
            return "South"
        elif orientation == "South":
            return "West"
        elif orientation == "West":
            return "North"
        elif orientation == "North":
            return "East"
    elif action == 4:
        if orientation == "East":
            return "North"
        elif orientation == "South":
            return "East"
        elif orientation == "West":
            return "South"
        elif orientation == "North":
            return "West"
    else:
        return orientation

for step in steps:
    with open(step, "r") as f:
        sample = json.load(f)
    try:
        action_history_text(sample["ground_truth_action"], action_mapping)
    except Exception as e:
        print(sample["ground_truth_action"])
        continue
    data.append({
        "id": i,
        "subtask": sample["text_instruction"],
        "orientation": reveal_position(sample["current_orientation"], sample["ground_truth_action"][0]),
        "target_orientation": sample["target_orientation"],
        "distance": str(int(sample["distance2target"] / 100)),
        "history": action_history_text(sample["past_actions"], action_mapping),
        "current_view": f"{os.path.split(step)[0]}/{sample['current_observation']}",
        "expected_view": f"{os.path.split(step)[0]}/{sample['visual_instruction']}",
        "action": action_mapping[sample["ground_truth_action"][0]],
        "plan": action_history_text(sample["ground_truth_action"], action_mapping)
    })
    i += 1
    
dataset = Dataset.from_list(data, features=features)
splits = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="action")
dataset = DatasetDict({
    "train": splits["train"],
    "validation": splits["test"]
})
repo_id = "Jise/citynav"
create_repo(repo_id, repo_type="dataset", exist_ok=True)
dataset.push_to_hub(repo_id)