import os
import logging
from datetime import datetime

def load_env_setting(filename):
    try:
        with open(get_settingpath(filename)) as f:
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext == '.json':
                import json
                return json.load(f)
            else:
                raise ValueError(f'Unsupported file type: {file_ext}')
    except FileNotFoundError:
        raise FileNotFoundError(f"Settings file {filename} not found")
    except Exception as e:
        raise Exception(f"Error loading settings: {str(e)}")

def get_settingpath(filename):
    import simworld_gym
    gympath = os.path.dirname(simworld_gym.__file__)
    return os.path.join(gympath, 'envs', 'setting', filename)

def print_info(msg):
    pass
    # print(msg)

def create_log(env_name):
    logging.basicConfig(filename=f"{env_name}_action_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

def write_log(step, action):
    logging.info(f"Step: {step}, Action: {action}")

def load_config(config_path):
    """
    Load configuration from a file.
    Supports JSON, YAML, and TOML formats.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration data
    """
    try:
        import simworld_gym
        gympath = os.path.dirname(simworld_gym.__file__)
        config_path = os.path.join(gympath, 'config', config_path)
        with open(config_path, 'r') as f:
            file_ext = os.path.splitext(config_path)[1].lower()
            if file_ext == '.json':
                import json
                return json.load(f)
            else:
                raise ValueError(f'Unsupported config file type: {file_ext}')
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {str(e)}")
        raise

def create_experiment_dir(log_dir, env_name):
    # Create timestamp-based experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(
        log_dir,
        f"{env_name}_{timestamp}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir