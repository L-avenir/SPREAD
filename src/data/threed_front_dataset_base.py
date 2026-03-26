import numpy as np

from functools import lru_cache
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset

import os
import pickle
from .utils_text import compute_loc_rel, reverse_rel, rotate_rel


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class DatasetDecoratorBase(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def bounds(self):
        return self._dataset.bounds

    @property
    def n_classes(self):
        return self._dataset.n_classes

    @property
    def class_labels(self):
        return self._dataset.class_labels

    @property
    def class_frequencies(self):
        return self._dataset.class_frequencies

    @property
    def n_object_types(self):
        return self._dataset.n_object_types

    @property
    def object_types(self):
        return self._dataset.object_types

    @property
    def feature_size(self):
        return self.bbox_dims + self.n_classes

    @property
    def bbox_dims(self):
        raise NotImplementedError()


    @property
    def n_predicate_types(self):
        return self._dataset.n_predicate_types
    
    @property
    def n_region_types(self):
        return self._dataset.n_region_types

    @property
    def predicate_types(self):
        return self._dataset.predicate_types

    @property
    def max_length(self):
        return self._dataset.max_length 


    def post_process(self, s):
        return self._dataset.post_process(s)


class BoxOrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, box_ordering=None):
        super().__init__(dataset)
        self.box_ordering = box_ordering

    @lru_cache(maxsize=16)
    def _get_boxes(self, scene_idx):
        scene = self._dataset[scene_idx]
        if self.box_ordering is None:
            return scene.bboxes
        elif self.box_ordering == "class_frequencies":
            return scene.ordered_bboxes_with_class_frequencies(
                self.class_frequencies
            )
        else:
            raise NotImplementedError()


class DataEncoder(BoxOrderedDataset):
    @property
    def property_type(self):
        raise NotImplementedError()


class RoomLayoutEncoder(DataEncoder):
    @property
    def property_type(self):
        return "room_layout"

    def __getitem__(self, idx):
        img = self._dataset[idx].room_mask[:, :, 0:1]
        return np.transpose(img, (2, 0, 1))

    @property
    def bbox_dims(self):
        return 0


class ClassLabelsEncoder(DataEncoder):
    @property
    def property_type(self):
        return "class_labels"

    def __getitem__(self, idx):
        classes = self.class_labels

        boxes = self._get_boxes(idx)
        L = len(boxes)
        C = len(classes)
        class_labels = np.zeros((L, C), dtype=np.float32)
        for i, bs in enumerate(boxes):
            class_labels[i] = bs.one_hot_label(classes)
        return class_labels

    @property
    def bbox_dims(self):
        return 0


class TranslationEncoder(DataEncoder):
    @property
    def property_type(self):
        return "translations"

    def __getitem__(self, idx):
        scene = self._dataset[idx]
        boxes = self._get_boxes(idx)
        L = len(boxes)
        translations = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            translations[i] = bs.centroid(-scene.centroid)
        return translations

    @property
    def bbox_dims(self):
        return 3


class SizeEncoder(DataEncoder):
    @property
    def property_type(self):
        return "sizes"

    def __getitem__(self, idx):
        boxes = self._get_boxes(idx)
        L = len(boxes)
        sizes = np.zeros((L, 3), dtype=np.float32)
        for i, bs in enumerate(boxes):
            sizes[i] = bs.size
        return sizes

    @property
    def bbox_dims(self):
        return 3


class AngleEncoder(DataEncoder):
    @property
    def property_type(self):
        return "angles"

    def __getitem__(self, idx):
        boxes = self._get_boxes(idx)
        L = len(boxes)
        angles = np.zeros((L, 1), dtype=np.float32)
        for i, bs in enumerate(boxes):
            angles[i] = bs.z_angle
        return angles

    @property
    def bbox_dims(self):
        return 1


class DatasetCollection(DatasetDecoratorBase):
    def __init__(self, *datasets):
        super().__init__(datasets[0])
        self._datasets = datasets

    @property
    def bbox_dims(self):
        return sum(d.bbox_dims for d in self._datasets)

    def __getitem__(self, idx):
        sample_params = {}
        for di in self._datasets:
            sample_params[di.property_type] = di[idx]
        return sample_params

    @staticmethod
    def collate_fn(samples):
        key_set = set(samples[0].keys()) - set(["length"])

        max_length = max(sample["length"] for sample in samples)

        padding_keys = set(k for k in key_set if len(samples[0][k].shape) == 2)
        sample_params = {}
        sample_params.update({
            k: np.stack([sample[k] for sample in samples], axis=0)
            for k in (key_set-padding_keys)
        })

        sample_params.update({
            k: np.stack([
                np.vstack([
                    sample[k],
                    np.zeros((max_length-len(sample[k]), sample[k].shape[1]))
                ]) for sample in samples
            ], axis=0)
            for k in padding_keys
        })
        sample_params["lengths"] = np.array([
            sample["length"] for sample in samples
        ])

        torch_sample = {
            k: torch.from_numpy(sample_params[k]).float()
            for k in sample_params
        }

        torch_sample.update({
            k: torch_sample[k][:, None]
            for k in torch_sample.keys()
            if "_tr" in k
        })

        return torch_sample


class CachedDatasetCollection(DatasetCollection):
    def __init__(self, dataset):
        super().__init__(dataset)
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset.get_room_params(idx)

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class Rotation(DatasetDecoratorBase):
    def __init__(self, dataset, min_rad=0.174533, max_rad=5.06145, fixed=False):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad
        self._fixed   = fixed

    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)
        else:
            return 0.0

    @property
    def fixed_rot_angle(self):
        if np.random.rand() < 0.25:
            return np.pi * 1.5
        elif np.random.rand() < 0.50:
            return np.pi
        elif np.random.rand() < 0.75:
            return np.pi * 0.5
        else:
            return 0.0

    def __getitem__(self, idx):
        if self._fixed:
            rot_angle = self.fixed_rot_angle
        else:
            rot_angle = self.rot_angle
        R = Rotation.rotation_matrix_around_y(rot_angle)

        sample_params = self._dataset[idx]
        sample_params["aug_angle"] = rot_angle
        for k, v in sample_params.items():
            if k == "translations":
                sample_params[k] = v.dot(R)

            elif k == "angles":
                angle_min, _ = self.bounds["angles"]
                sample_params[k] = \
                    (v + rot_angle - angle_min) % (2 * np.pi) + angle_min

            elif k == "room_layout":
                img = np.transpose(v, (1, 2, 0))
                sample_params[k] = np.transpose(rotate(
                    img, rot_angle * 180 / np.pi, reshape=False
                ), (2, 0, 1))

        return sample_params
    
class CASTRotation(DatasetDecoratorBase):
    def __init__(self, dataset, axis):
        super().__init__(dataset)
        
        self.axis = axis

    def __getitem__(self, idx):
        rot_angle = np.random.uniform(-np.pi, np.pi)
        
        cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
        if self.axis == 'y':
            R = np.array([[ cos_a, 0, sin_a],
                            [     0, 1,     0],
                            [-sin_a, 0, cos_a]])
        elif self.axis == 'z':
            R = np.array([[ cos_a, sin_a, 0],
                            [-sin_a, cos_a,     0],
                            [0, 0, 1]])

        sample_params = self._dataset[idx]

        if "translations" in sample_params:
            sample_params["translations"] = (R @ sample_params["translations"].T).T

        if "angles" in sample_params:
            sample_params["angles"] = R @ sample_params["angles"]

        return sample_params

class Scale(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k in bounds:
                print(f"Postprocess [{k}] by bounds {bounds[k][0]}~{bounds[k][1]}")
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 1


class Scale_CosinAngle(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                sample_params[k] = np.concatenate([np.cos(v), np.sin(v)], axis=-1)

            elif k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k == "angles":
                print(f"Postprocess [{k}] by [arctan2]")
                sample_params[k] = np.arctan2(v[..., 1:2], v[..., 0:1])

            elif k in bounds:
                print(f"Postprocess [{k}] by bounds {bounds[k][0]}~{bounds[k][1]}")
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 2
    
class Scale_Rot6d(DatasetDecoratorBase):
    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x


    def __getitem__(self, idx):
        bounds = self.bounds
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "angles":
                rot6d = v[:, :, :2].transpose(0, 2, 1).reshape(v.shape[0], 6)
                sample_params[k] = rot6d

            elif k in bounds:
                sample_params[k] = Scale.scale(
                    v, bounds[k][0], bounds[k][1]
                )
        return sample_params

    def post_process(self, sample_params):
        bounds = self.bounds
        for k, v in sample_params.items():
            if k == "angles":
                sample_params[k] = rot6d_to_rotmat(v)

            elif k in bounds:
                sample_params[k] = Scale.descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return super().post_process(sample_params)

    @property
    def bbox_dims(self):
        return 3 + 3 + 6


class Jitter(DatasetDecoratorBase):
    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles"]:
                sample_params[k] = v + np.random.normal(0, 0.01)
        return sample_params


class Permutation(DatasetDecoratorBase):
    def __init__(self, dataset, permutation_keys, permutation_axis=0):
        super().__init__(dataset)
        self._permutation_keys = permutation_keys
        self._permutation_axis = permutation_axis

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        shapes = sample_params["translations"].shape
        ordering_except_first = np.random.permutation(shapes[self._permutation_axis] - 1) + 1
        ordering = np.concatenate([[0], ordering_except_first])
        sample_params["permutation"] = ordering

        for k in self._permutation_keys:
            if k not in sample_params or sample_params[k] is None:
                continue


            if k == "relations":
                if sample_params[k].shape[0] > 0:
                    idx_mapping = {ordering[i]: i for i in range(len(ordering))}
                    sample_params[k][:, 0] = np.vectorize(idx_mapping.get)(sample_params[k][:, 0])
                    sample_params[k][:, 2] = np.vectorize(idx_mapping.get)(sample_params[k][:, 2])
            elif k == "descriptions":
                sample_params[k]["obj_class_ids"] = [sample_params[k]["obj_class_ids"][i] for i in ordering]
                idx_mapping = {ordering[i]: i for i in range(len(ordering))}
                for i in range(len(sample_params[k]["obj_relations"])):
                    s, p, o = sample_params[k]["obj_relations"][i]
                    s_new, o_new = idx_mapping[s], idx_mapping[o]
                    sample_params[k]["obj_relations"][i] = (s_new, p, o_new)
            elif k == "regions":
                sample_params[k] = sample_params[k][ordering[:,None], ordering]
                

            else:
                if k in ['meshes']:
                    sample_params[k] = [sample_params[k][i] for i in ordering]
                else:
                    sample_params[k] = sample_params[k][ordering]

        return sample_params


class OrderedDataset(DatasetDecoratorBase):
    def __init__(self, dataset, ordered_keys, box_ordering=None):
        super().__init__(dataset)
        self._ordered_keys = ordered_keys
        self._box_ordering = box_ordering

    def __getitem__(self, idx):
        if self._box_ordering is None:
            return self._dataset[idx]

        if self._box_ordering != "class_frequencies":
            raise NotImplementedError()

        sample = self._dataset[idx]
        order = self._get_class_frequency_order(sample)
        for k in self._ordered_keys:
            sample[k] = sample[k][order]
        return sample

    def _get_class_frequency_order(self, sample):
        t = sample["translations"]
        c = sample["class_labels"].argmax(-1)
        class_frequencies = self.class_frequencies
        class_labels = self.class_labels
        f = np.array([
            [class_frequencies[class_labels[ci]]]
            for ci in c
        ])

        return np.lexsort(np.hstack([t, f]).T)[::-1]


class Add_GlobalRegionGraph(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)
        
        self.region_keys = np.array(list(range(9)))

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        translations = sample_params["translations"][1:]
        N = translations.shape[0]
        
        diff = translations[None, :, :] - translations[:, None, :]

        pos_x = diff[:, :, 0] > 0
        pos_y = diff[:, :, 1] > 0
        pos_z = diff[:, :, 2] > 0

        regions = (pos_x.astype(np.int32) << 2) | (pos_y.astype(np.int32) << 1) | pos_z.astype(np.int32)

        np.fill_diagonal(regions, 8)
        
        sample_params["regions"][:,:] = 8
        sample_params["regions"][1:, 1:] = regions
        
        return sample_params


class Add_SceneGraph(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]

        if not os.path.exists(sample_params["relation_path"]):
            relations = []
            num_objs = len(sample_params["translations"])

            for idx in range(num_objs):
                c1_id = sample_params["class_labels"][idx, :].argmax()
                t1 = sample_params["translations"][idx, :]
                r1 = sample_params["angles"][idx, 0]
                s1 = sample_params["sizes"][idx, :]
                corners1 = trs_to_corners(t1, r1, s1)
                name1 = self.object_types[c1_id]

                for other_idx in range(idx+1, num_objs):
                    c2_id = sample_params["class_labels"][other_idx, :].argmax()
                    t2 = sample_params["translations"][other_idx, :]
                    r2 = sample_params["angles"][other_idx, 0]
                    s2 = sample_params["sizes"][other_idx, :]
                    corners2 = trs_to_corners(t2, r2, s2)
                    name2 = self.object_types[c2_id]

                    loc_rel_str = compute_loc_rel(corners1, corners2, name1, name2)
                    if loc_rel_str is not None:
                        relation_id = self.predicate_types.index(loc_rel_str)
                        relations.append([idx, relation_id, other_idx])


            sample_params["relations"] = np.array(relations)
            if "aug_angle" not in sample_params or sample_params["aug_angle"] == 0.:
                np.save(sample_params["relation_path"], sample_params["relations"])

        else:
            sample_params["relations"] = np.load(sample_params["relation_path"], allow_pickle=True)

            if "aug_angle" in sample_params:
                for i in range(len(sample_params["relations"])):
                    p = sample_params["relations"][i, 1]
                    sample_params["relations"][i, 1] = self.predicate_types.index(
                        rotate_rel(self.predicate_types[p], sample_params["aug_angle"])
                    )

        return sample_params


class Add_Description(DatasetDecoratorBase):
    def __init__(self, dataset, only_describe_main_objects=True, seed=None):
        super().__init__(dataset)
        self.only_describe_main_objects = only_describe_main_objects
        self.seed = seed

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        assert "relations" in sample_params, "Relations must be computed before adding descriptions."

        if not os.path.exists(sample_params["description_path"]):
            descriptions = {
                "obj_class_ids": [],
                "obj_relations": [],
            }

            class_ids = sample_params["class_labels"].argmax(axis=1)
            descriptions["obj_class_ids"] = class_ids.tolist()

            if self.only_describe_main_objects:
                relations = {}
                for i in range(len(sample_params["relations"])):
                    s, p, o = sample_params["relations"][i]
                    if p != self.n_predicate_types:
                        rev_p = self.predicate_types.index(
                            reverse_rel(self.predicate_types[p])
                        )
                        try:
                            relations[s].append((p, o))
                        except KeyError:
                            relations[s] = [(p, o)]
                        try:
                            relations[o].append((rev_p, s))
                        except KeyError:
                            relations[o] = [(rev_p, s)]

                obj_volumes = np.array(list(map(lambda x: x[0]*x[2], sample_params["sizes"])))
                obj_volumn_sorted_indices = np.argsort(obj_volumes)[::-1]
                main_obj_indices = obj_volumn_sorted_indices[:min(3, len(obj_volumn_sorted_indices))]

                relation_ids = []
                for s in main_obj_indices:
                    if relations.get(s) is not None:
                        for p, o in relations[s]:
                            relation_ids.append((
                                int(s), int(p), int(o)
                            ))
                descriptions["obj_relations"] = relation_ids

            else:
                relation_ids = []
                for triples in sample_params["relations"]:
                    s, p, o = triples
                    relation_ids.append((
                        int(s), int(p), int(o)
                    ))
                descriptions["obj_relations"] = relation_ids

            if "aug_angle" not in sample_params or sample_params["aug_angle"] == 0.:
                with open(sample_params["description_path"], 'wb') as f:
                    pickle.dump(descriptions, f)
            sample_params["descriptions"] = descriptions

        else:
            with open(sample_params["description_path"], 'rb') as f:
                descriptions = pickle.load(f)

            if "aug_angle" in sample_params:
                for i in range(len(descriptions["obj_relations"])):
                    s_class_id, p, o_class_id = descriptions["obj_relations"][i]
                    descriptions["obj_relations"][i] = (
                        int(s_class_id),
                        int(self.predicate_types.index(
                            rotate_rel(self.predicate_types[p], sample_params["aug_angle"])
                        )),
                        int(o_class_id)
                    )
            sample_params["descriptions"] = descriptions

        return sample_params


class Add_Objfeature(DatasetDecoratorBase):
    def __init__(self, dataset, objfeat_type="openshape_vitg14"):
        super().__init__(dataset)
        self.objfeat_type = objfeat_type

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        sample_params[f"{self.objfeat_type}_features"] = \
            np.load(sample_params[f"{self.objfeat_type}_path"])

        return sample_params
    
class Add_Uni3Dfeature(DatasetDecoratorBase):
    def __init__(self, dataset, uni3d_encoder):
        super().__init__(dataset)
        
        self.device = 'cuda'
        self.model = uni3d_encoder

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        
        pc = torch.tensor(sample_params['xyz'], dtype=torch.float32).to(device=self.device, non_blocking=True)
        rgb = torch.tensor(sample_params['rgb'], dtype=torch.float32).to(device=self.device, non_blocking=True)
        feature = torch.cat((pc, rgb),dim=-1)

        pc_features = utils.get_model(self.model).encode_pc(feature) 
        sample_params['pc_features'] = pc_features.detach().cpu().numpy()
        
        return sample_params





def trs_to_corners(t: np.ndarray, r: float, s: np.ndarray) -> np.ndarray:
    template = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [ 1, -1, -1], [ 1, -1, 1], [ 1, 1, -1], [ 1, 1, 1]
    ])
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(r)
    R[0, 2] = -np.sin(r)
    R[2, 0] = np.sin(r)
    R[2, 2] = np.cos(r)
    R[1, 1] = 1.

    return (template * s).dot(R) + t

def rot6d_to_rotmat(x):
    a1 = x[..., :3]
    a2 = x[..., 3:6]
    
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    
    b3 = np.cross(b1, b2, axis=-1)
    
    rotmat = np.stack([b1, b2, b3], axis=-1)
    return rotmat

