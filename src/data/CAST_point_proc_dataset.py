import csv
from collections import Counter, OrderedDict
from functools import lru_cache
import numpy as np
import json
import os

from PIL import Image

from .common import BaseDataset
from .threed_front_scene import Room
from .utils import parse_threed_front_scenes

import random
from transformers import AutoImageProcessor, AutoModel

from .utils_mesh import Mesh
import trimesh

import torch


class ThreedFront(BaseDataset):
    def __init__(self, scenes, bounds=None):
        super().__init__(scenes)
        assert isinstance(self.scenes[0], Room)
        self._object_types = None
        self._room_types = None
        self._count_furniture = None
        self._bbox = None

        self._sizes = self._centroids = self._angles = None
        if bounds is not None:
            self._sizes = bounds["sizes"]
            self._centroids = bounds["translations"]
            self._angles = bounds["angles"]

            if "openshape_vitg14" in bounds:
                self._openshape_vitg14 = bounds["openshape_vitg14"]

        else:
            self._openshape_vitg14 = None

        self._max_length = None

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self.scenes), self.n_object_types
        )

    @property
    def bbox(self):
        if self._bbox is None:
            _bbox_min = np.array([1000, 1000, 1000])
            _bbox_max = np.array([-1000, -1000, -1000])
            for s in self.scenes:
                bbox_min, bbox_max = s.bbox
                _bbox_min = np.minimum(bbox_min, _bbox_min)
                _bbox_max = np.maximum(bbox_max, _bbox_max)
            self._bbox = (_bbox_min, _bbox_max)
        return self._bbox

    def _centroid(self, box, offset):
        return box.centroid(offset)

    def _size(self, box):
        return box.size

    def _compute_bounds(self):
        _size_min = np.array([10000000]*3)
        _size_max = np.array([-10000000]*3)
        _centroid_min = np.array([10000000]*3)
        _centroid_max = np.array([-10000000]*3)
        _angle_min = np.array([10000000000])
        _angle_max = np.array([-10000000000])

        _openshape_vitg14_min = np.array([10000000])
        _openshape_vitg14_max = np.array([-10000000])
        all_openshape_vitg14 = []

        for s in self.scenes:
            for f in s.bboxes:
                if np.any(f.size > 5):
                    print(s.scene_id, f.size, f.model_uid, f.scale)
                centroid = self._centroid(f, -s.centroid)
                _centroid_min = np.minimum(centroid, _centroid_min)
                _centroid_max = np.maximum(centroid, _centroid_max)
                _size_min = np.minimum(self._size(f), _size_min)
                _size_max = np.maximum(self._size(f), _size_max)
                _angle_min = np.minimum(f.z_angle, _angle_min)
                _angle_max = np.maximum(f.z_angle, _angle_max)


                if f.openshape_vitg14_features is not None:
                    all_openshape_vitg14.append(f.openshape_vitg14_features)


        self._sizes = (_size_min, _size_max)
        self._centroids = (_centroid_min, _centroid_max)
        self._angles = (_angle_min, _angle_max)


        if len(all_openshape_vitg14) > 0:
            all_openshape_vitg14 = np.stack(all_openshape_vitg14, axis=0)
            _openshape_vitg14_min, _openshape_vitg14_max = np.array([all_openshape_vitg14.min()]), np.array([all_openshape_vitg14.max()])
            self._openshape_vitg14 = (_openshape_vitg14_min, _openshape_vitg14_max)


    @property
    def bounds(self):
        bounds = {
            "translations": self.centroids,
            "sizes": self.sizes,
            "angles": self.angles
        }




        return bounds

    @property
    def sizes(self):
        if self._sizes is None:
            self._compute_bounds()
        return self._sizes

    @property
    def centroids(self):
        if self._centroids is None:
            self._compute_bounds()
        return self._centroids

    @property
    def angles(self):
        if self._angles is None:
            self._compute_bounds()
        return self._angles


    @property
    def openshape_vitg14(self):
        if self._openshape_vitg14 is None:
            self._compute_bounds()
        return self._openshape_vitg14


    @property
    def count_furniture(self):
        if self._count_furniture is None:
            counts = []
            for s in self.scenes:
                counts.append(s.furniture_in_room)
            counts = Counter(sum(counts, []))
            counts = OrderedDict(sorted(counts.items(), key=lambda x: -x[1]))
            self._count_furniture = counts
        return self._count_furniture

    @property
    def class_order(self):
        return dict(zip(
            self.count_furniture.keys(),
            range(len(self.count_furniture))
        ))

    @property
    def class_frequencies(self):
        object_counts = self.count_furniture
        class_freq = {}
        n_objects_in_dataset = sum(
            [object_counts[k] for k, v in object_counts.items()]
        )
        for k, v in object_counts.items():
            class_freq[k] = object_counts[k] / n_objects_in_dataset
        return class_freq

    @property
    def object_types(self):
        if self._object_types is None:
            self._object_types = set()
            for s in self.scenes:
                self._object_types |= set(s.object_types)
            self._object_types = sorted(self._object_types)
        return self._object_types


    @property
    def predicate_types(self):
        return ["Empty", "Support", "Contact"]
    
    @property
    def region_types(self):
        return ["right", "left", "in front of", "behind", "above", "below", "self"]


    @property
    def room_types(self):
        if self._room_types is None:
            self._room_types = set([s.scene_type for s in self.scenes])
        return self._room_types

    @property
    def class_labels(self):
        return self.object_types + ["start", "end"]


    @property
    def max_length(self):
        if self._max_length is None:
            _room_types = set([str(s.scene_type) for s in self.scenes])
            if 'bed' in _room_types:
                self._max_length = 12
            elif 'living' in _room_types:
                self._max_length = 21
            elif 'dining' in _room_types:
                self._max_length = 21
            elif 'library' in _room_types:
                self._max_length = 11

        return self._max_length


    @classmethod
    def from_dataset_directory(cls, dataset_directory, path_to_model_info,
                               path_to_models, path_to_room_masks_dir=None,
                               path_to_bounds=None, filter_fn=lambda s: s):
        scenes = parse_threed_front_scenes(
            dataset_directory,
            path_to_model_info,
            path_to_models,
            path_to_room_masks_dir
        )
        bounds = None
        if path_to_bounds:
            bounds = np.load(path_to_bounds, allow_pickle=True)

        return cls([s for s in map(filter_fn, scenes) if s], bounds)


class CachedRoom(object):
    def __init__(
        self,
        scene_id,
        room_layout,
        floor_plan_vertices,
        floor_plan_faces,
        floor_plan_centroid,
        class_labels,
        translations,
        sizes,
        angles,
        image_path
    ):
        self.scene_id = scene_id
        self.room_layout = room_layout
        self.floor_plan_faces = floor_plan_faces
        self.floor_plan_vertices = floor_plan_vertices
        self.floor_plan_centroid = floor_plan_centroid
        self.class_labels = class_labels
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.image_path = image_path

    @property
    def floor_plan(self):
        return np.copy(self.floor_plan_vertices), \
            np.copy(self.floor_plan_faces)

    @property
    def room_mask(self):
        return self.room_layout[:, :, None]
    
class CachedScene(object):
    def __init__(
        self,
        scene_tag,
        xyz,
        rgb,
        translations,
        sizes,
        angles,
        relations,
        obj_names
    ):
        self.scene_tag = scene_tag
        self.translations = translations
        self.sizes = sizes
        self.angles = angles
        self.xyz = xyz
        self.rgb = rgb
        self.relations = relations
        self.obj_names = obj_names

    @property
    def floor_plan(self):
        return np.copy(self.floor_plan_vertices), \
            np.copy(self.floor_plan_faces)

    @property
    def room_mask(self):
        return self.room_layout[:, :, None]


class Cached_CAST_Point_Proc_Dataset(ThreedFront):
    def __init__(self, base_dir, config, split="train"):
        self._base_dir = base_dir
        self.config = config
        self.split = split[0]
        
        self.geometry_mode = config.get("geometry_mode", "voxel")
        
        dinov2_model_name="facebook/dinov2-base"
        self.processor = AutoImageProcessor.from_pretrained(dinov2_model_name)


        self._max_length = self.config.get("max_length", None)

        print(f"Loading data from {self._base_dir}, split = {self.split}")

        with open(os.path.join(self._base_dir, "train_test_split.json"), "r") as f:
            self.split_map = json.load(f)[self.split]

        self._tags = []
        for group in os.listdir(self._base_dir):
            if os.path.isdir(os.path.join(self._base_dir, group)):
                tag = group
                if tag in self.split_map:
                    self._tags.append(tag)
                                
        self._path_to_scenes = sorted([
            os.path.join(self._base_dir, pi, "boxes.npz")
            for pi in self._tags
        ])
        
        if self.geometry_mode == 'mesh':
            self.assets_folder = "./dataset/meshes"
            
            with np.load(os.path.join(self.voxels_folder, "voxels.npz")) as data:
                voxel_mask = data["voxel_mask"]
                norm_scale = data["normalization_scale_factor"]
            self.voxels = torch.from_numpy(voxel_mask).to(torch.uint8).to('cuda')
            self.normalize_factor = torch.from_numpy(norm_scale).to('cuda')
            
            with open(os.path.join(self.voxels_folder, "voxel_file_order.json"), "r") as f:
                self.voxel_file_order = json.load(f)
                
            self.voxel_file_order = {k:i for i, k in enumerate(self.voxel_file_order)}
        
        
        self._parse_train_stats(config["train_stats"])
    
    def is_valid_scene(self, scene):
        if not os.path.exists(scene):
            return False
        
        data = np.load(scene)
        
        if str(self.split_map[str(data['scene_id'])]) not in self.split:
            return False
        
        if str(data['scene_id']) in self.invalid_scene_ids:
            return False

        if len(data['jids']) < 3 or len(data['jids']) > 12:
            return False
        
        if  any(j in self.invalid_bbox_jids for j in data['jids']):
            return False
        
        return True
    
    def compute_minmax(self):
        size_min = np.full(3, np.inf)
        size_max = np.full(3, -np.inf)
        T_min = np.full(3, np.inf)
        T_max = np.full(3, -np.inf)

        for path in self._path_to_scenes:
            data = np.load(path)
            
            size = data['size']
            T = data['T_bbox']

            size_min = np.minimum(size_min, size.min(axis=0))
            size_max = np.maximum(size_max, size.max(axis=0))

            T_min = np.minimum(T_min, T[1:].min(axis=0))
            T_max = np.maximum(T_max, T[1:].max(axis=0))

        print(f"Size min per axis: {size_min}")
        print(f"Size max per axis: {size_max}")
        print(f"T min per axis: {T_min}")
        print(f"T max per axis: {T_max}")
        
        self.size_min = size_min
        self.size_max = size_max
        self.T_min = T_min
        self.T_max = T_max


    @lru_cache(maxsize=32)
    def __getitem__(self, i):
        D = np.load(self._path_to_scenes[i])
        if 'bedroom' in self._path_to_scenes[i]:
            return CachedScene(
                scene_tag=self._path_to_scenes[i],
                xyz=np.concatenate([np.zeros_like(D["xyz"][:1,:,:]), D["xyz"]], axis=0),
                rgb=D["rgb"],
                translations=D["T"],
                sizes=np.repeat(D["scale_factor"].reshape(-1,1), 3, axis=-1),
                factor=np.repeat(D["origin_scale_factor"].reshape(-1,1), 3, axis=-1),
                angles=D["R"],
                relations=D["relations"],
                regios=D["regions"],
                uni3d_feature=D['uni3d_feature'],
                clip_feature=D['clip_feature'],
                miche_feature=D['miche_feature'],
                obj_names=D["obj_names"],
                jids=np.array([""] + list(D["jids"]))
            )
        else:
            return CachedScene(
                scene_tag=self._path_to_scenes[i],
                xyz=np.concatenate([np.zeros_like(D["xyz"][:1]), D["xyz"]], axis=0),
                rgb=D["rgb"],
                translations=D["T"],
                sizes=D["size"],
                angles=D["R"],
                relations=D["relations"],
                regios=D["regions"],
                uni3d_feature=D['uni3d_feature'],
                clip_feature=D['clip_feature'],
                miche_feature=D['miche_feature'],
                obj_names=D["obj_names"]
            )

    def get_room_params(self, i):
        boxes_path = self._path_to_scenes[i]
        D = np.load(boxes_path)
        
        image_folder = os.path.join(os.path.dirname(boxes_path), "multiview_render")
        if os.path.exists(image_folder):
            images = sorted([i for i in os.listdir(image_folder) if i.endswith(".jpg")])
            image = random.choice(images)
            cam_id = int(image.split("_")[1].split(".")[0])
            image_path = os.path.join(image_folder, image)
            image = Image.open(image_path).convert("RGB")
            image = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0).numpy()
            cam_npz = np.load(os.path.join(image_folder, "image_data.npz"))
            cam = {'extrinsic': cam_npz['camera_poses'][cam_id],
                'intrinsic': cam_npz['camera_intrinsics'][cam_id]}
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            image_path = None
            cam = None
        
        meshes = None
        voxels = None
        voxel_normalize_factor = None
        xyz = None
        normal = None
        if self.geometry_mode == 'mesh':
            meshes = [None]
            for id, name in enumerate(D['obj_names']):
                if id == 0:
                    continue
                
                mesh_path = os.path.join(self.assets_folder, f"{name}_unpacked.glb")
                if os.path.exists(mesh_path):
                    mesh = trimesh.load(os.path.join(self.assets_folder, f"{name}_unpacked.glb"), force='mesh')
                    mesh.apply_translation(-D['bbox_center'][id])
                    mesh = Mesh(mesh.vertices, mesh.faces)
                    meshes.append(mesh)
        elif self.geometry_mode == 'voxel':
            voxel_ids = [self.voxel_file_order[f"{name}_unpacked.npz"] for name in D['obj_names'][1:]]
            voxels = self.voxels[voxel_ids]
            voxel_normalize_factor = self.normalize_factor[voxel_ids]
            
            voxels = torch.cat([torch.zeros_like(voxels[:1]), voxels], dim=0)
            voxel_normalize_factor = torch.cat([torch.zeros_like(voxel_normalize_factor[:1]), voxel_normalize_factor], dim=0)
            xyz = D["xyz"]
            idx = np.random.choice(xyz.shape[1], 2000, replace=False)
            xyz = xyz[:, idx, :]
        elif self.geometry_mode == 'point':
            xyz = D["xyz"]
            normal = D["normal"]
            idx = np.random.choice(xyz.shape[1], 1000, replace=False)
            xyz = xyz[:, idx, :]
            normal = normal[:, idx, :]
        else:
            pass
        
        if 'clip_feature' in D:
            clip_feature = D['clip_feature']
        else:
            clip_feature = np.zeros_like(D["T_bbox"])
        return {
            'scene_tag':self._path_to_scenes[i],
            'translations':D["T_bbox"],
            'bbox_center':D["bbox_center"],
            'sizes':D["size"],
            'angles':D["R_bbox"],
            'relations':D["relations"],
            'regions':D["regions_bbox"],
            'obj_names':D["obj_names"],
            'clip_feature':clip_feature,
            'miche_feature':D["miche_feature"],
            'image': image,
            'image_path': image_path,
            'camera': cam,
            'meshes': meshes,
            'voxels': voxels,
            'voxel_normalize_factors': voxel_normalize_factor,
            'xyz': xyz,
            'normal': normal
            }

    def __len__(self):
        return len(self._path_to_scenes)

    def __str__(self):
        return "Dataset contains {} scenes with {} discrete types".format(
                len(self), self.n_object_types
        )
        
    def write_data_state_txt(self, state_path):
        state = {'bounds_translations': np.concatenate(self._centroids, axis=-1).reshape(6).tolist(),
                 'bounds_sizes': np.concatenate(self._sizes, axis=-1).reshape(6).tolist(),
                 'bounds_angles': []
                 }
        with open(state_path, "w") as f:
            json.dump(state, f)

    def _parse_train_stats(self, train_stats):
        state_path = os.path.join(self._base_dir, train_stats)
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                train_stats = json.load(f)
            self._centroids = train_stats["bounds_translations"]
            self._centroids = (
                np.array(self._centroids[:3]),
                np.array(self._centroids[3:])
            )
            self._sizes = train_stats["bounds_sizes"]
            self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
            self._angles = []
            
        else:
            self.compute_minmax()
            self._centroids = (
                self.T_min,
                self.T_max
            )
            self._sizes = (self.size_min, self.size_max)
            self._angles = []
            
            self.write_data_state_txt(state_path)
            





    @property
    def predicate_types(self):
        return ["Empty", "Support", "Contact"]
    
    @property
    def region_types(self):
        if "globalregion" in self.config["encoding_type"]:
            return ["right, in front of, below",
                    "right, in front of, above",
                    "right, behind, below",
                    "right, behind, above",
                    "left, in front of, below",
                    "left, in front of, above",
                    "left, behind, below",
                    "left, behind, above",
                    "self"]
        else:
            return ["right", "left", "in front of", "behind", "above", "below", "self"]

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture
