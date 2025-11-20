from glob import glob
import os
import pandas as pd
import json
import re
from metric import *
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

run_names = ["simple/gpt-4o-mini", "simple/gpt-5-mini"]

for run_name in run_names:
    sr = 0
    spl_v = 0
    subtask_sr_v = 0
    intersection_sr_v = 0
    distance_v = 0
    distance_progress_v = 0
    scollision_v = 0
    dcollision_v = 0
    violation_v = 0
    ndtw_v = 0
    i = 0
    
    paths = glob(f"log_{run_name}/logs/simpleenv_*/reset_*/trajectory.csv")

    task_success = []

    for log_path in paths:
        i += 1
        for p in log_path.split("/"):
            if p.startswith("reset_"):
                break
        map_dir = "_".join(p.split("_")[2:6])
        task_dir = "_".join(p.split("_")[6:])
        task_template_dir = os.path.join("SinWorld/task_generator/output/single_world/total/test_map", "easy_mode", map_dir, task_dir, "task_config.json")
        with open(task_template_dir, "r") as f:
            task_template = json.load(f)
        log_df = pd.read_csv(log_path)
        current_spl = spl(log_df, task_template)
        current_subtask_sr = subtask_sr(log_df, task_template)
        # current_ndtw = ndtw(log_df, task_template)
        spl_v += current_spl
        if current_spl > 0:
            sr += 1
        subtask_sr_v += current_subtask_sr
        # distance_v += distance(log_df, task_template)
        distance_progress_v += distance_progress(log_df, task_template)
        # intersection_sr_v += intersection_sr(log_df, task_template)
        # scollision_v += static_collision(log_df, task_template)
        # dcollision_v += dynamic_collision(log_df, task_template)
        # violation_v += violation(log_df, task_template)
        # ndtw_v += current_ndtw
        task_success.append(current_subtask_sr)
    print(f"Run Name: {run_name}, SR: {round(sr / i, 4)}, Average Subtask SR: {round(subtask_sr_v / i, 4)}, Average Distance Progress: {round(distance_progress_v / i, 4)}")