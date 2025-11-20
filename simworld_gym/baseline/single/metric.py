import pandas as pd
import json
import numpy as np

def spl(log_df, task_template):
    success = log_df["success"][len(log_df) - 1]
    distance = log_df["moving_distance"][len(log_df) - 1]
    ideal_path = task_template["ideal_path"]
    ideal_distance = 0
    for i in range(len(ideal_path) - 1):
        ideal_distance += np.linalg.norm(np.array(ideal_path[i]) - np.array(ideal_path[i+1]))
    if success:
        return min(ideal_distance / distance, 1)
    else:
        return 0

def intersection_success(log_df, task_template):
    current_point = np.array([log_df["agent_x"][0], log_df["agent_y"][0]])
    j = 1
    current_target = np.array(task_template["ideal_path"][j])
    subtask_success = 0
    for i in range(1, len(log_df)):
        current_point = np.array([log_df["agent_x"][i], log_df["agent_y"][i]])
        if np.linalg.norm(current_point - current_target) < 15:
            subtask_success += 1
            j += 1
            if j >= len(task_template["ideal_path"]):
                break
            current_target = np.array(task_template["ideal_path"][j])
    return subtask_success

def intersection_sr(log_df, task_template):
    current_point = np.array([log_df["agent_x"][0], log_df["agent_y"][0]])
    j = 1
    current_target = np.array(task_template["ideal_path"][j])
    subtask_success = 0
    for i in range(1, len(log_df)):
        current_point = np.array([log_df["agent_x"][i], log_df["agent_y"][i]])
        if np.linalg.norm(current_point - current_target) < 15:
            subtask_success += 1
            j += 1
            if j >= len(task_template["ideal_path"]):
                break
            current_target = np.array(task_template["ideal_path"][j])
    return subtask_success / (len(task_template["ideal_path"]) - 1)

def subtask_success(log_df, task_template):
    return log_df["subtask_success"][len(log_df) - 1]
    
def subtask_sr(log_df, task_template):
    return log_df["subtask_success"][len(log_df) - 1] / len(task_template["instruction"])

def distance(log_df, task_template):
    final_point = np.array([log_df["agent_x"][len(log_df) - 1], log_df["agent_y"][len(log_df) - 1]])
    target_point = np.array([log_df["target_x"][len(log_df) - 1], log_df["target_y"][len(log_df) - 1]])
    return np.linalg.norm(final_point - target_point)

def distance_progress(log_df, task_template):
    all_points = [np.array([log_df["agent_x"][i], log_df["agent_y"][i]]) for i in range(len(log_df))]
    target_point = np.array([log_df["target_x"][len(log_df) - 1], log_df["target_y"][len(log_df) - 1]])
    initial_distance = np.linalg.norm(all_points[0] - target_point)
    closest_distance = np.min([np.linalg.norm(point - target_point) for point in all_points])
    return max(0, (initial_distance - closest_distance) / initial_distance)

def ndtw(log_df, task_template, d=1000):
    current_point = np.array([log_df["agent_x"][0], log_df["agent_y"][0]])
    agent_path = [current_point]
    for i in range(1, len(log_df)):
        new_point = np.array([log_df["agent_x"][i], log_df["agent_y"][i]])
        if np.linalg.norm(current_point - new_point) > 1e-2:
            agent_path.append(new_point)
            current_point = new_point
    ideal_path = np.array(task_template["ideal_path"])
    interpolated_path = []
    for i in range(len(ideal_path) - 1):
        p1 = ideal_path[i]
        p2 = ideal_path[i+1]
        length = np.linalg.norm(p1 - p2)
        n_points = int(np.floor(length / d))
        for j in range(n_points):
            t = (j * d) / length
            interpolated_path.append(p1 + t * (p2 - p1))
    interpolated_path.append(ideal_path[-1])
    n, m = len(agent_path), len(interpolated_path)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            dist = np.linalg.norm(agent_path[i-1] - interpolated_path[j-1])
            dtw[i, j] = dist + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    dtw = dtw[n, m]
    return np.exp(-dtw / m / d)
    

def static_collision(log_df, task_template):
    return sum(log_df["object_collision"]) + sum(log_df["building_collision"])

def dynamic_collision(log_df, task_template):
    return sum(log_df["human_collision"]) + sum(log_df["vehicle_collision"])

def violation(log_df, task_template):
    return sum(log_df["red_light_violation"])


if __name__ == "__main__":
    log_dir = "log.csv"
    template_dir = "task_config.json"
    log_df = pd.read_csv(log_dir)
    print(log_df)
    with open(template_dir, "r") as f:
        task_template = json.load(f)
    print(spl(log_df, task_template))
    print(ndtw(log_df, task_template))