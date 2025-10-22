import unreal
import json
import os

class AssetExtractor:
    def __init__(self, save_directory, file_name):
        """
        Initialize the AssetExtractor with a save directory and file name.

        :param save_directory: Directory where the JSON file will be saved.
        :param file_name: Name of the JSON file.
        """
        self.save_path = os.path.join(save_directory, file_name)

    def get_selected_assets(self):
        """
        Extracts selected assets from the Unreal Editor and returns them as a list of nodes.

        :return: List of nodes representing selected assets.
        """
        selected_assets = unreal.EditorLevelLibrary.get_selected_level_actors()
        nodes = []

        for index, asset in enumerate(selected_assets):
            location = asset.get_actor_location()
            rotation = asset.get_actor_rotation()
            bounds_origin, bounds_extent = asset.get_actor_bounds(only_colliding_components=True, include_from_child_actors=True)

            node = {
                "id": index,
                "instance_name": asset.get_class().get_name(),
                "properties": {
                    "location": {
                        "x": location.x,
                        "y": location.y,
                        "z": location.z
                    },
                    "orientation": {
                        "roll": round(rotation.roll, 2),
                        "pitch": round(rotation.pitch, 2),
                        "yaw": round(rotation.yaw, 2)
                    },
                    "dimensions": {
                        "length": round(bounds_extent.x, 2),
                        "width": round(bounds_extent.y, 2)
                    }
                }
            }
            nodes.append(node)
        return nodes

    def save_to_json(self, nodes):
        """
        Saves the nodes to a JSON file.

        :param nodes: List of nodes to be saved.
        """
        json_data = {
            "base_map": {
                "name": "map_1",
                "env_bin": "gym_citynav\\Binaries\\Win64\\gym_citynav.exe",
                "width": 11000.0,
                "height": 12750.0
            },
            "nodes": nodes
        }

        with open(self.save_path, 'w') as file:
            json.dump(json_data, file, indent=2)

# Usage
save_directory = r"C:\Users\mjkim\OneDrive\Documents\Unreal Projects\gym_citynav\SCAI-CityNav\gym_citynav\gym_citynav\envs\setting\simple_world"
file_name = "simple_world_world.json"

extractor = AssetExtractor(save_directory, file_name)
nodes = extractor.get_selected_assets()
extractor.save_to_json(nodes) 