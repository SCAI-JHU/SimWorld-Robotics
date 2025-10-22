import unrealcv
from unrealcv.util import read_png, read_npy
import cv2
import time
import PIL.Image
from io import BytesIO
import numpy as np
import json
import os
from simworld.communicator.unrealcv import UnrealCV as SimworldUnrealCV

class UnrealCV(SimworldUnrealCV):
    def __init__(self, port, ip, resolution):
        super().__init__(port, ip, resolution)


    def set_location_hard(self, loc, name):
        [x, y, z] = loc
        cmd = f'vset /object/{name}/location {x} {y} {z}'
        with self.lock:
            self.client.request(cmd)

    def custom_set_orientation(self, orientation, name):
        [roll, pitch, yaw] = orientation
        # unreal's rotation order is pitch, yaw, roll
        cmd = f'vset /object/{name}/rotation {pitch} {yaw} {roll}'
        with self.lock:
            self.client.request(cmd)

    def destroy_hard(self, actor_name):
        # print(f"Destroying {actor_name}")
        cmd = f'vset /object/{actor_name}/destroy'
        with self.lock:
            self.client.request(cmd)

    def get_total_collision(self, name):
        with self.lock:
            res = self.client.request(f'vbp {name} GetCollisionNum')
        human_collision = eval(json.loads(res)["HumanCollision"])
        object_collision = eval(json.loads(res)["ObjectCollision"])
        building_collision = eval(json.loads(res)["BuildingCollision"])
        vehicle_collision = eval(json.loads(res)["VehicleCollision"])
        return human_collision, object_collision, building_collision, vehicle_collision


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
        [roll, pitch, yaw] = orientation
        # unreal's rotation order is pitch, yaw, roll
        cmd = f'vset /object/{name}/rotation {pitch} {yaw} {roll}'
        with self.lock:
            self.client.request(cmd)
        
    def get_available_batch(self, actor_names):
        cmd = [f'vbp {actor_name} CheckAvailable' for actor_name in actor_names]
        with self.lock:
            res = self.client.request_batch(cmd)
        # Parse JSON response and convert to float
        return [float(json.loads(r)["IsAvailable"]) > 0 for r in res]

    def get_cameras(self):
        cmd = "vget /cameras"
        with self.lock:
            res = self.client.request(cmd)
        cameras = res.strip().split(" ")
        return cameras

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
                    with self.lock:
                        image = self._decode_npy(self.client.request(cmd))
                else:
                    cmd = f'vget /camera/{cam_id}/{viewmode} png'
                    # image = read_png(self.client.request(cmd))
                    with self.lock:
                        image = self._decode_png(self.client.request(cmd))
            elif mode == 'file':  # save image to file and read it
                img_path = os.path.join(os.getcwd(), f"{cam_id}-{viewmode}.png")
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                with self.lock:
                    img_dirs = self.client.request(cmd)
                image = cv2.imread(img_dirs)

            elif mode == 'fast':  # get image from unrealcv in bmp format
                cmd = f'vget /camera/{cam_id}/{viewmode} bmp'
                with self.lock:
                    image = self._decode_bmp(self.client.request(cmd))
            
            elif mode == 'file_path':  # save image to file and read it
                cmd = f'vget /camera/{cam_id}/{viewmode} {img_path}'
                with self.lock:
                    img_dirs = self.client.request(cmd)
                image = read_png(img_dirs)

            if image is None:
                raise ValueError(f"Failed to read image with mode={mode}, viewmode={viewmode}")
            # print(f"read_image: image={image}")
            return image

        except Exception as e:
            print(f"Error reading image: {str(e)}")
            return np.zeros((480, 640, 3), dtype=np.uint8)  # 可以根据需要调整尺寸

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
