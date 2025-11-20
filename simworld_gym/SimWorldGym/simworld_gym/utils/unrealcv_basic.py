from unrealcv.util import read_png, read_npy
import cv2
import time
import PIL.Image
from io import BytesIO
import numpy as np
import json
import os
import ast
from simworld.communicator.unrealcv import UnrealCV as SimworldUnrealCV


def _parse_collision_value(raw):
    if isinstance(raw, (int, float)):
        return raw
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0


class UnrealCV(SimworldUnrealCV):
    def __init__(self, port, ip, resolution):
        super().__init__(port, ip)
        self.resolution = resolution
        self.ini_unrealcv(resolution)

    def ini_unrealcv(self, resolution=(320, 240)):
        self.check_connection()
        [w, h] = resolution
        # self.client.request(f'vrun setres {w}x{h}w', -1)  # set resolution of the display window
        # self.client.request('DisableAllScreenMessages', -1)  # disable all screen messages
        # self.client.request('vrun sg.ShadowQuality 1', -1)  # set shadow quality to low
        # self.client.request('vrun sg.TextureQuality 1', -1)  # set texture quality to low
        # self.client.request('vrun sg.EffectsQuality 1', -1)  # set effects quality to low

        self.client.request('vrun Editor.AsyncSkinnedAssetCompilation 2', -1)

        time.sleep(1.0)
        
        self.client.message_handler = self.message_handler

    def message_handler(self, message):
        msg = message

    def check_connection(self):
        while self.client.isconnected() is False:
            print('UnrealCV server is not running. Please try again')
            time.sleep(1)
            self.client.connect()

    def spawn(self, prefab, name):
        cmd = f'vset /objects/spawn {prefab} {name}'
        self.client.request(cmd)

    def spawn_bp_asset(self, prefab_path, name):
        cmd = f'vset /objects/spawn_bp_asset {prefab_path} {name}'
        self.client.request(cmd)

    def set_location(self, loc, name):
        [x, y, z] = loc
        cmd = f'vset /object/{name}/location {x} {y} {z}'
        self.client.request(cmd)
    
    def set_location_hard(self, loc, name):
        [x, y, z] = loc
        cmd = f'vset /object/{name}/location {x} {y} {z}'
        with self.lock:
            self.client.request(cmd)

    def set_orientation(self, orientation, name):
        """Unified method to set orientation (replaces custom_set_orientation)"""
        [roll, pitch, yaw] = orientation
        # unreal's rotation order is pitch, yaw, roll
        cmd = f'vset /object/{name}/rotation {pitch} {yaw} {roll}'
        with self.lock:
            self.client.request(cmd)

    def set_scale(self, scale, name):
        [x, y, z] = scale
        cmd = f'vset /object/{name}/scale {x} {y} {z}'
        self.client.request(cmd)

    def enable_controller(self, name, enable_controller):
        cmd = f'vbp {name} EnableController {enable_controller}'
        self.client.request(cmd)

    def set_physics(self, actor_name, hasPhysics):
        cmd = f'vset /object/{actor_name}/physics {hasPhysics}'
        self.client.request(cmd)

    def set_collision(self, actor_name, hasCollision):
        cmd = f'vset /object/{actor_name}/collision {hasCollision}'
        self.client.request(cmd)

    def set_movable(self, actor_name, isMovable):
        cmd = f'vset /object/{actor_name}/object_mobility {isMovable}'
        self.client.request(cmd)
    
    def set_color(self, actor_name, color):
        [R, G, B] = color
        cmd = f'vset /object/{actor_name}/color {R} {G} {B}'
        self.client.request(cmd)

    def destroy(self, actor_name):
        # print(f"Destroying {actor_name}")
        cmd = f'vset /object/{actor_name}/destroy'
        self.client.request(cmd)
    
    def destroy_hard(self, actor_name):
        # print(f"Destroying {actor_name}")
        cmd = f'vset /object/{actor_name}/destroy'
        with self.lock:
            self.client.request(cmd)

    def apply_action_transition(self, robot_name, action):
        [speed, duration, direction] = action
        if speed < 0:
            # switch the direction
            if direction == 0:
                direction = 1
            elif direction == 1:
                direction = 0
            elif direction == 2:
                direction = 3
            elif direction == 3:
                direction = 2
        cmd = f'vbp {robot_name} Move_Speed {speed} {duration} {direction}'
        self.client.request(cmd)

    def apply_action_rotation(self, robot_name, action):
        [duration, angle, direction] = action
        if direction == -1 and angle > 0:
            angle = -angle
        cmd = f'vbp {robot_name} Rotate_Angle {duration} {angle} {direction}'
        self.client.request(cmd)

    def apply_action_look_up(self, robot_name):
        cmd = f'vbp {robot_name} lookup'
        self.client.request(cmd)
    
    def apply_action_look_down(self, robot_name):
        cmd = f'vbp {robot_name} lookdown'
        self.client.request(cmd)
    
    def get_objects(self):
        res = self.client.request('vget /objects')
        objects = np.array(res.split())
        return objects

    def get_total_collision(self, name):
        cmd = f'vbp {name} GetCollisionNum'
        with self.lock:
            res = self.client.request(cmd)
        payload = json.loads(res)
        human_collision = _parse_collision_value(payload.get("HumanCollision", 0))
        object_collision = _parse_collision_value(payload.get("ObjectCollision", 0))
        building_collision = _parse_collision_value(payload.get("BuildingCollision", 0))
        vehicle_collision = _parse_collision_value(payload.get("VehicleCollision", 0))
        return human_collision, object_collision, building_collision, vehicle_collision

    def get_total_collision_batch(self, actor_names):
        if not actor_names:
            return []
        cmd = [f'vbp {actor_name} GetCollisionNum' for actor_name in actor_names]
        responses = self.client.request_batch(cmd)
        collisions = []
        for res in responses:
            payload = json.loads(res)
            collisions.append((
                _parse_collision_value(payload.get("HumanCollision", 0)),
                _parse_collision_value(payload.get("ObjectCollision", 0)),
                _parse_collision_value(payload.get("BuildingCollision", 0)),
                _parse_collision_value(payload.get("VehicleCollision", 0)),
            ))
        return collisions

    def get_location(self, actor_name):
        cmd = f'vget /object/{actor_name}/location'
        res = self.client.request(cmd)
        location = [float(i) for i in res.split()]
        return np.array(location)

    def get_location_batch(self, actor_names):
        cmd = [f'vget /object/{actor_name}/location' for actor_name in actor_names]
        with self.lock:
            res = self.client.request_batch(cmd)
        # Parse each response and convert to numpy array
        locations = [np.array([float(i) for i in r.split()]) for r in res]
        return locations

    def get_is_available(self, actor_name):
        cmd = f'vbp {actor_name} CheckAvailable'
        with self.lock:
            res = self.client.request(cmd)
        # Parse JSON response and convert to float
        is_available = float(json.loads(res)["IsAvailable"])
        if is_available == 0:
            return False
        elif is_available == 1:
            return True
        else:
            raise ValueError(f"Failed to get the availability of the actor {actor_name}")
        
    def set_orientation_hard(self, orientation, name):
        """Alias for backward compatibility"""
        self.set_orientation(orientation, name)
        
    def get_available_batch(self, actor_names):
        cmd = [f'vbp {actor_name} CheckAvailable' for actor_name in actor_names]
        with self.lock:
            res = self.client.request_batch(cmd)
        # Parse JSON response and convert to float
        return [float(json.loads(r)["IsAvailable"]) > 0 for r in res]
    
    def get_orientation(self, actor_name):
        cmd = f'vget /object/{actor_name}/rotation'
        res = self.client.request(cmd)
        location = [float(i) for i in res.split()]
        return np.array(location)

    def get_orientation_batch(self, actor_names):
        cmd = [f'vget /object/{actor_name}/rotation' for actor_name in actor_names]
        res = self.client.request_batch(cmd)
        # Parse each response and convert to numpy array
        orientations = [np.array([float(i) for i in r.split()]) for r in res]
        return orientations

    def get_cameras(self):
        cmd = "vget /cameras"
        with self.lock:
            res = self.client.request(cmd)
        cameras = res.strip().split(" ")
        return cameras

    def clean_garbage(self):
        self.client.request("vset /action/clean_garbage")

    def show_img(self, img, title="raw_img"):
        cv2.imshow(title, img)
        cv2.waitKey(3)

    def set_fps(self, fps):
        cmd = f'vset /action/set_fixed_frame_rate {fps}'
        with self.lock:
            self.client.request(cmd)
    
    def read_image(self, cam_id, viewmode, mode='direct', img_path=None):
        # cam_id:0 1 2 ...
        # viewmode:lit, normal, depth, object_mask
        # mode: direct, file，file_path
        image = None
        # print(f"read_image: cam_id={cam_id}, viewmode={viewmode}, mode={mode}")
        try:
            if mode == 'direct':  # get image from unrealcv in png format
                if viewmode == 'depth':
                    cmd = f'vget /camera/{cam_id}/{viewmode} npy'
                    # image = read_npy(self.client.request(cmd))
                    image = self._decode_npy(self.client.request(cmd))
                else:
                    cmd = f'vget /camera/{cam_id}/{viewmode} png'
                    # image = read_png(self.client.request(cmd))
                    image = self._decode_png(self.client.request(cmd))
            elif mode == 'file':  # save image to file and read it
                img_path = os.path.join(os.getcwd(), f"{cam_id}-{viewmode}.png")
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)

            elif mode == 'fast':  # get image from unrealcv in bmp format
                cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
                image = self._decode_bmp(self.client.request(cmd))
            
            elif mode == 'file_path':  # save image to file and read it
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                img_dirs = self.client.request(cmd)
                image = read_png(img_dirs)

            if image is None:
                raise ValueError(f"Failed to read image with mode={mode}, viewmode={viewmode}")
            return image

        except Exception as e:
            print(f"Error reading image: {str(e)}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def _decode_npy(self, res): # decode npy image
        image = np.load(BytesIO(res))
        eps = 1e-6
        depth_log = np.log(image + eps)

        depth_min = np.min(depth_log)
        depth_max = np.max(depth_log)
        normalized_depth = (depth_log - depth_min) / (depth_max - depth_min)

        gamma = 0.5
        normalized_depth = np.power(normalized_depth, gamma)

        image = (normalized_depth * 255).astype(np.uint8)

        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        return image

    def _decode_png(self, res): # decode png image
        img = np.asarray(PIL.Image.open(BytesIO(res)))
        img = img[:, :, :-1]  # delete alpha channel
        img = img[:, :, ::-1]  # transpose channel order
        return img

    def _decode_bmp(self, res, channel=4): # decode bmp image
        # TODO: Error reading image: cannot reshape array of size 1228858 into shape (960,1280,4)
        img = np.fromstring(res, dtype=np.uint8)
        img=img[-self.resolution[1]*self.resolution[0]*channel:]
        img=img.reshape(self.resolution[1], self.resolution[0], channel)
        return img[:, :, :-1] # delete alpha channel

    def get_observations(self, cam_id, mode='direct'):
        images = []
        for viewmode in ['lit', 'normal', 'depth', 'object_mask']:
            image = self.read_image(cam_id, viewmode, mode)
            images.append(image)
        return images

    def export_observation(self, cam_id, mode='direct'):
        images = self.get_observations(cam_id, mode)
        output_dir = 'observation'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        for viewmode, image in zip(['lit', 'normal', 'depth', 'object_mask'], images):
            # ndarrag to png
            image = PIL.Image.fromarray(image)
            output_path = os.path.join(output_dir, f'{timestamp}_{cam_id}_{viewmode}_{mode}.png')
            image.save(output_path)
        return images
