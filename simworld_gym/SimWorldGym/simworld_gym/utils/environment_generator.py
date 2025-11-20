from simworld_gym.utils import misc
import pandas as pd
import numpy as np

ENV_ASSET_PATH = "asset/AssetPath.json"

class EnvironmentGenerator(object):
    def __init__(self, unrealcv):
        self.node_df = None
        self.unrealcv = unrealcv
        self.asset_library = None
        self.generated_ids = set()

    def _parse_rgb(self, color_str):
        """Parse RGB values from color string like '(R=255,G=255,B=0)'"""
        import re
        pattern = r'R=(\d+),G=(\d+),B=(\d+)'
        match = re.search(pattern, color_str)
        if match:
            return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
        return [0, 0, 0]  # Default to black if parsing fails

    def loading_json(self, world_json):
        # load world from json
        world_setting = misc.load_env_setting(world_json)
        # use pandas data structure, convert json data into pandas data frame
        nodes = world_setting['nodes']
        node_df = pd.json_normalize(nodes, sep='_')
        node_df.set_index('id', inplace=True)
        self.node_df = node_df
        # load asset library
        self.asset_library = misc.load_config(ENV_ASSET_PATH)


    def generate_world(self):
        def process_node(row):
            # spawn every node on the map
            id = row.name  # name is the index of the row
            try:
                instance_ref = self.asset_library[self.node_df.loc[id, "instance_name"]]["asset_path"]
                color = self.asset_library["colors"][self.asset_library[self.node_df.loc[id, "instance_name"]]["color"]]
                rgb_values = self._parse_rgb(color)
            except KeyError:
                print("Can't find node {} in asset library".format(self.node_df.loc[id, "instance_name"]))
                pass
            else:
                self.unrealcv.spawn_bp_asset(instance_ref, id)
                # self.unrealcv.set_color(id, rgb_values)
                location = self.node_df.loc[id, ['properties_location_x', 'properties_location_y', 'properties_location_z']].to_list()
                self.unrealcv.set_location(location, id)
                orientation = self.node_df.loc[id, ['properties_orientation_roll', 'properties_orientation_pitch', 'properties_orientation_yaw']].to_list()
                self.unrealcv.set_orientation(orientation, id)
                scale = self.node_df.loc[id, ['properties_scale_x', 'properties_scale_y', 'properties_scale_z']].to_list()
                self.unrealcv.set_scale(scale, id)
                self.unrealcv.set_collision(id, True)
                self.unrealcv.set_movable(id, False)
                self.generated_ids.add(id)

        # self.unrealcv.set_color("Floor", self._parse_rgb(self.asset_library["colors"]['FLOOR']))
        # self.unrealcv.set_color("StaticMeshActor_0", self._parse_rgb(self.asset_library["colors"]['SKY']))
        self.node_df.apply(process_node, axis=1)

    def clear_env(self, has_road: bool = True):
        # Get all the objects in the environment
        objects = [obj.lower() for obj in self.unrealcv.get_objects()]  # Convert objects to lowercase
        # Define unwanted objects
        if has_road:
            unwanted_terms = ['GEN_BP_', 'GEN_Road_']
        else:
            unwanted_terms = ['GEN_BP_']
        unwanted_terms = [term.lower() for term in unwanted_terms]  # Convert unwanted terms to lowercase

        # Get all the objects starting with the name in unwanted_terms
        indexes = np.concatenate([np.flatnonzero(np.char.startswith(objects, term)) for term in unwanted_terms])
        # Destroy them
        if indexes is not None:
            for index in indexes:
                self.unrealcv.destroy(objects[index])

        self.unrealcv.clean_garbage()


    # append new assets for node_df without covering original assets
    def append_loading_json(self, world_json):
        new_world_setting = misc.load_env_setting(world_json)
        new_nodes = new_world_setting['nodes']
        new_node_df = pd.json_normalize(new_nodes, sep='_')
        new_node_df.set_index('id', inplace=True)
        if self.node_df is None:
            self.node_df = new_node_df
        else:
            new_ids = new_node_df.index.difference(self.node_df.index)
            if not new_ids.empty:
                self.node_df = pd.concat([self.node_df, new_node_df.loc[new_ids]])
        
    # generate the assets lastly added
    def generate_new_world(self):
        def process_new_node(row):
            node_id = row.name
            # if the node had been generated, skip it
            if node_id in self.generated_ids:
                return
            try:
                instance_ref = self.asset_library[self.node_df.loc[node_id, "instance_name"]]["asset_path"]
                print(instance_ref)
            except KeyError:
                print("Can't find node {} in asset library".format(node_id))
            else:
                self.unrealcv.spawn_bp_asset(instance_ref, node_id)
                location = self.node_df.loc[node_id, ['properties_location_x', 'properties_location_y', 'properties_location_z']].to_list()
                self.unrealcv.set_location(location, node_id)
                orientation = self.node_df.loc[node_id, ['properties_orientation_roll', 'properties_orientation_pitch', 'properties_orientation_yaw']].to_list()
                self.unrealcv.set_orientation(orientation, node_id)
                scale = self.node_df.loc[node_id, ['properties_scale_x', 'properties_scale_y', 'properties_scale_z']].to_list()
                self.unrealcv.set_scale(scale, node_id)
                self.unrealcv.set_collision(node_id, True)
                self.unrealcv.set_movable(node_id, False)
                # mark the generated node
                self.generated_ids.add(node_id)

        # select the nodes that haven't been generated
        new_nodes = self.node_df.loc[~self.node_df.index.isin(self.generated_ids)]
        new_nodes.apply(process_new_node, axis=1)

