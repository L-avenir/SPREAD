from .utils_text import reverse_rel
from .threed_front_dataset_base import *

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time

def box2transform(box, scene_id, obj_id):
    T = box["translations"][scene_id][obj_id]
    size = box["sizes"][scene_id][obj_id]
    R = box["angles"][scene_id][obj_id]
    
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T
    
    return transform

def check_overlaps(meshes):
    bboxs = [mesh.bounding_box() for mesh in meshes]
    bboxs_np = np.array(bboxs)
    centers = (bboxs_np[:, :3] + bboxs_np[:, 3:]) / 2.0
    sizes = bboxs_np[:, 3:] - bboxs_np[:, :3]
    diag_lengths = np.linalg.norm(sizes, axis=1)
    radii = diag_lengths / 2.0
    min_distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2) - (radii[:, None] + radii[None, :])
    overlaps = min_distances < 0.0
    np.fill_diagonal(overlaps, False)
    overlap_pairs = np.argwhere(overlaps)
    overlap_pairs = overlap_pairs[overlap_pairs[:, 0] < overlap_pairs[:, 1]]
    overlap_pairs = overlap_pairs.tolist()
    return overlap_pairs


def check_overlaps_aabb_np(meshes):
    bboxes = np.array([m.bounding_box() for m in meshes], dtype=float)
    mins = bboxes[:, :3]
    maxs = bboxes[:, 3:]

    overlaps_dim = (mins[:, None, :] <= maxs[None, :, :]) & (mins[None, :, :] <= maxs[:, None, :])

    overlaps = overlaps_dim.all(axis=2)

    np.fill_diagonal(overlaps, False)

    i_idx, j_idx = np.triu_indices(overlaps.shape[0], k=1)

    mask = overlaps[i_idx, j_idx]
    overlap_pairs = np.stack((i_idx[mask], j_idx[mask]), axis=1)

    return overlap_pairs.tolist()

class SG2SC(DatasetDecoratorBase):
    def __init__(self, dataset, objfeat_type=None, Uni3D_encoder=None, mode='train', diffusion_type="ddpm"):
        super().__init__(dataset)
        self.objfeat_type = objfeat_type
        self.Uni3D_encoder = Uni3D_encoder
        self.device = 'cuda'
        self.mode = mode

        if diffusion_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="linear",
                variance_type="fixed_small",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        elif diffusion_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        else:
            raise NotImplementedError
        
    def __len__(self):
        return super().__len__()
    
    def compute_pc_features(self, meshes, bbox, num_points=2000):
        bbox = bbox.unsqueeze(0)
        bbox_params = {
                        "translations": bbox[..., :3].cpu().numpy(),
                        "sizes": bbox[..., 3:6].cpu().numpy(),
                        "angles": bbox[..., 6:].cpu().numpy()
                        }
        boxes_pred = self.post_process(bbox_params)
        
        B, N, _ = bbox.shape
        
        pc_features = []
        
        meshes = meshes[1:]
            
        for obj_id, mesh in enumerate(meshes):
            mesh.apply_transform(box2transform(boxes_pred, 0, obj_id+1))
            
        overlap_pairs = check_overlaps_aabb_np(meshes)
        
        points = np.array([mesh.sample_points(num_points) for mesh in meshes])
        inside = np.zeros((len(meshes), num_points, 1), dtype=bool)
        
        for pair in overlap_pairs:
            mesh1 = meshes[pair[0]]
            mesh2 = meshes[pair[1]]
            
            points1 = points[pair[0]]
            points2 = points[pair[1]]
            
            sdf1 = mesh2.compute_sdf(points1).numpy().reshape(-1, 1)
            sdf2 = mesh1.compute_sdf(points2).numpy().reshape(-1, 1)
            
            inside[pair[0]] = np.logical_or(sdf1 < 0, inside[pair[0]])
            inside[pair[1]] = np.logical_or(sdf2 < 0, inside[pair[1]])
            
        points = torch.tensor(points, dtype=torch.float32).to(bbox.device)
        inside = torch.tensor(inside, dtype=torch.float32).to(bbox.device)
        pc_feature = torch.cat([points, inside], dim=-1)
        pc_feature_padded = torch.zeros((N, num_points, 4), dtype=pc_feature.dtype, device=pc_feature.device)
        pc_feature_padded[1:1+pc_feature.shape[0]] = pc_feature
        pc_features.append(pc_feature_padded)
            
        pc_features = torch.stack(pc_features, dim=0)

        return pc_features

    def __getitem__(self, idx):
        start_time = time.time()
        
        sample_params = self._dataset[idx]
        sample_params["length"] = sample_params["translations"].shape[0]
        return sample_params

    def collate_fn(self, samples):
        sample_params_pad = {
            "scene_uids": [],
            "boxes": [],
            "clip_feature": [],
            "miche_feature": [],
            "edges": [],
            "regions": [],
            "obj_masks": [],
            "obj_names": [],
            "image": [],
            "image_path": [],
            "camera": [],
            "bbox_center": [],
        }

        max_length = max(sample["length"] for sample in samples)
        
        for k in ['voxels', 'voxel_normalize_factors', 'meshes', 'xyz', 'normal']:
            if samples[0].get(k) is not None:
                sample_params_pad[k] = []

        for sample_params in samples:
            
            scene_uid = str(sample_params["scene_tag"])
            obj_names = sample_params["obj_names"].tolist()
            bbox_center = sample_params["bbox_center"]
            triples = sample_params["relations"]
            regions = sample_params["regions"]
            image = sample_params["image"]
            image_path = sample_params["image_path"]
            camera = sample_params["camera"]
            
            
            miche_feature = sample_params["miche_feature"]
            clip_feature = sample_params["clip_feature"]
            
            if 'voxels' in sample_params_pad:
                voxels = sample_params["voxels"]
                voxel_normalize_factors = sample_params["voxel_normalize_factors"]
                sample_params_pad["voxels"].append(voxels)
                sample_params_pad["voxel_normalize_factors"].append(voxel_normalize_factors)
            
            if 'meshes' in sample_params_pad:
                sample_params_pad['meshes'].append(sample_params['meshes'])
                
            if 'xyz' in sample_params_pad:
                xyz = sample_params["xyz"]
                normal = sample_params["normal"]
                sample_params_pad["xyz"].append(xyz)
            if 'normal' in sample_params_pad:
                sample_params_pad["normal"].append(normal)

            sample_params_pad["scene_uids"].append(scene_uid)
            sample_params_pad["obj_names"].append(obj_names)
            sample_params_pad["bbox_center"].append(bbox_center)
            sample_params_pad["image"].append(image)
            sample_params_pad["image_path"].append(image_path)
            sample_params_pad["camera"].append(camera)

            boxes = np.concatenate([
                sample_params["translations"],
                sample_params["sizes"],
                sample_params["angles"]
            ], axis=-1)
            
            sample_params_pad["boxes"].append(np.pad(
                boxes, ((0, max_length - boxes.shape[0]), (0, 0)),
                mode="constant", constant_values=0.
            ))
            
            sample_params_pad["clip_feature"].append(np.pad(
                clip_feature, ((0, max_length - clip_feature.shape[0]), (0, 0)),
                mode="constant", constant_values=0.
            ))
            sample_params_pad["miche_feature"].append(np.pad(
                miche_feature, ((0, max_length - miche_feature.shape[0]), (0, 0), (0, 0)),
                mode="constant", constant_values=0.
            ))

            edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)
            for s, p, o in triples:

                edges[s, o] = p
                
            sample_params_pad["edges"].append(edges)
            
            padded_regions = self.n_region_types * np.ones((max_length, max_length), dtype=np.int64)
            n = regions.shape[0]
            padded_regions[:n, :n] = regions
            sample_params_pad["regions"].append(padded_regions)

            obj_mask = np.zeros(max_length, dtype=np.int64)
            obj_mask[:sample_params["length"]] = 1
            sample_params_pad["obj_masks"].append(obj_mask)
            
        for k, v in sample_params_pad.items():
            if k in ["scene_uids", "obj_names", "bbox_center", "jids", "factor", "origin_scale", "image_path", "camera", "meshes"]:
                sample_params_pad[k] = v
            elif k in ["bbox", "boxes", "noise", "noisy_boxes", 'xyz', 'normal', 'rgb', "pc_features", "clip_feature", "miche_feature", "image"]:
                sample_params_pad[k] = torch.from_numpy(np.stack(v, axis=0)).float()
            elif k in ["voxels", "voxel_normalize_factors"]:
                sample_params_pad[k] = torch.stack(v, dim=0)      
            else:
                sample_params_pad[k] = torch.from_numpy(np.stack(v, axis=0)).long()
               
    
        return sample_params_pad

    @property
    def bbox_dims(self):
        return self._dataset.bbox_dims


class SGDiffusion(DatasetDecoratorBase):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, idx):
        sample_params = self._dataset[idx]
        max_length = self.max_length

        sample_params_new = {}
        for k, v in sample_params.items():
            if k in ["translations", "sizes", "angles"]:
                p = np.copy(v)
                L, C = p.shape
                sample_params_new[k] = np.vstack([p, np.tile(np.zeros(C)[None, :], [max_length - L, 1])]).astype(np.float32)

            elif k == "class_labels":
                class_labels = np.copy(v)
                new_class_labels = np.concatenate([class_labels[:, :-2], class_labels[:, -1:]], axis=-1)
                L, C = new_class_labels.shape
                end_label = np.eye(C)[-1]
                sample_params_new["objs"] = np.vstack([
                    new_class_labels, np.tile(end_label[None, :], [max_length - L, 1])
                ]).argmax(axis=-1)
                sample_params_new["length"] = L

            elif k == "relations":
                triples = np.copy(v)
                edges = self.n_predicate_types * np.ones((max_length, max_length), dtype=np.int64)
                for s, p, o in triples:
                    rev_p = self.predicate_types.index(
                        reverse_rel(self.predicate_types[p])
                    )
                    edges[s, o] = p
                    edges[o, s] = rev_p
                uppertri_edges = edges[np.triu_indices(max_length, k=1)]
                assert uppertri_edges.shape[0] == max_length * (max_length - 1) // 2
                sample_params_new["edges"] = uppertri_edges

        sample_params_new["scene_uid"] = sample_params["scene_uid"]

        if "descriptions" in sample_params:
            sample_params_new["descriptions"] = sample_params["descriptions"]

        with open(sample_params["models_info_path"], "rb") as f:
            models_info = pickle.load(f)
        objfeat_vq_indices = [
            np.array(model_info["objfeat_vq_indices"])
            for model_info in models_info
        ]
        object_descs = [
            model_info["chatgpt_caption"]
            for model_info in models_info
        ]
        if "permutation" in sample_params:
            objfeat_vq_indices = [objfeat_vq_indices[i] for i in sample_params["permutation"]]
            object_descs = [object_descs[i] for i in sample_params["permutation"]]

        objfeat_vq_indices = np.vstack(objfeat_vq_indices)
        objfeat_vq_indices_pad = 64 * np.ones([max_length, objfeat_vq_indices.shape[1]])
        objfeat_vq_indices_pad[:objfeat_vq_indices.shape[0]] = objfeat_vq_indices
        sample_params_new["objfeat_vq_indices"] = objfeat_vq_indices_pad
        objfeats_vq = np.eye(64)[objfeat_vq_indices]
        objfeats_vq_pad = np.zeros([max_length, objfeats_vq.shape[1], objfeats_vq.shape[2]])
        objfeats_vq_pad[:objfeats_vq.shape[0]] = objfeats_vq
        sample_params_new["objfeats_vq"] = objfeats_vq_pad * 2. - 1.
        sample_params_new["object_descs"] = object_descs

        return sample_params_new

    def collate_fn(self, samples):
        sample_params_batch = {
            "scene_uids": [],
            "lengths": [],
            "objs": [],
            "edges": [],
            "boxes": [],
            "descriptions": [],
            "objfeat_vq_indices": [],
            "objfeats_vq": [],
            "object_descs": []
        }

        for sample_params in samples:
            scene_uid = str(sample_params["scene_uid"])
            length = sample_params["length"]
            objs = sample_params["objs"]
            edges = sample_params["edges"]
            boxes = np.concatenate([
                sample_params["translations"],
                sample_params["sizes"],
                sample_params["angles"]
            ], axis=-1)

            sample_params_batch["scene_uids"].append(scene_uid)
            sample_params_batch["lengths"].append(length)
            sample_params_batch["objs"].append(objs)
            sample_params_batch["edges"].append(edges)
            sample_params_batch["boxes"].append(boxes)

            if "descriptions" in sample_params:
                descriptions = sample_params["descriptions"]
                sample_params_batch["descriptions"].append(descriptions)

            objfeat_vq_indices = sample_params["objfeat_vq_indices"]
            sample_params_batch["objfeat_vq_indices"].append(objfeat_vq_indices.reshape(-1))
            objfeats_vq = sample_params["objfeats_vq"]
            sample_params_batch["objfeats_vq"].append(objfeats_vq.reshape(objfeats_vq.shape[0], -1))

            if "object_descs" in sample_params:
                object_descs = sample_params["object_descs"]
                sample_params_batch["object_descs"].append(object_descs)

        for k, v in sample_params_batch.items():
            if k in ["scene_uids", "descriptions", "object_descs"]:
                sample_params_batch[k] = v
            elif k in ["boxes", "room_masks"]:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).float()
            else:
                sample_params_batch[k] = torch.from_numpy(np.stack(v, axis=0)).long()

        return sample_params_batch




def dataset_encoding_factory(
    name,
    dataset,
    augmentations=None,
    box_ordering=None,
    Uni3D_encoder=None
) -> DatasetDecoratorBase:

    if "cached" in name:
        dataset_collection = OrderedDataset(
            CachedDatasetCollection(dataset),
            ["class_labels", "translations", "sizes", "angles"],
            box_ordering=box_ordering
        )
    else:
        box_ordered_dataset = BoxOrderedDataset(
            dataset,
            box_ordering
        )
        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)

        dataset_collection = DatasetCollection(
            class_labels,
            translations,
            sizes,
            angles
        )

    if name == "basic":
        return DatasetCollection(
            class_labels,
            translations,
            sizes,
            angles
        )

    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotation":
                print("Apply [rotation] augmentation")
                dataset_collection = Rotation(dataset_collection)
            elif aug_type == "global_rotation_y":
                print("Applying [global rotation] augmentation")
                dataset_collection = CASTRotation(dataset_collection, 'y')
            elif aug_type == "global_rotation_z":
                print("Applying [global rotation] augmentation")
                dataset_collection = CASTRotation(dataset_collection, 'z')
            elif aug_type == "fixed_rotation":
                print("Applying [fixed rotation] augmentation")
                dataset_collection = Rotation(dataset_collection, fixed=True)
            elif aug_type == "jitter":
                print("Apply [jittering] augmentation")
                dataset_collection = Jitter(dataset_collection)

    if "globalregion" in name:
        print("Add [global region graph] to the dataset")
        dataset_collection = Add_GlobalRegionGraph(dataset_collection)

    if "graph" in name or "desc" in name:
        print("Add [scene graphs] to the dataset")
        dataset_collection = Add_SceneGraph(dataset_collection)
        
    if "desc" in name:
        if "seed" in name:
            seed = int(name.split("_")[-1])
        else:
            seed = None
        print("Add [scene descriptions] to the dataset")
        dataset_collection = Add_Description(dataset_collection, seed=seed)

    objfeat_type = None
    if "objfeat" in name:
        print("Add [object features] to the dataset")
        if "openshape_vitg14" in name:
            objfeat_type = "openshape_vitg14"
        else:
            raise ValueError(f"Not found valid object feature type in [{name}]")
        dataset_collection = Add_Objfeature(dataset_collection, objfeat_type)

    print(f"Scale {list(dataset_collection.bounds.keys())}")
    if "sincos_angle" in name:
        print("Use [cos, sin] for angle encoding")
        dataset_collection = Scale_CosinAngle(dataset_collection)
    elif 'rot6d' in name:
        print("Use rot6d SO(3) for angle encoding")
        dataset_collection = Scale_Rot6d(dataset_collection)
    else:
        dataset_collection = Scale(dataset_collection)

    permute_keys = [
        "xyz", "normal", "translations", "sizes", "angles", "origin_scale", "meshes", "bbox_center"
        "relations", "regions", "uni3d_feature", "clip_feature", "miche_feature", "obj_names", "jids", "factor", "voxels", "voxel_normalize_factors"
    ]


    if "sg2sc" in name:
        print("Use [Sg2Sc diffusion] model")
        if "no_prm" in name or "eval" in name:
            return SG2SC(dataset_collection, objfeat_type, Uni3D_encoder, mode='eval')
        else:
            return SG2SC(dataset_collection, objfeat_type, Uni3D_encoder, mode='train')

    elif "sgdiffusion" in name:
        assert "graph" in name or "desc" in name, \
            "Add scene graphs to the dataset first (as ground-truth)."
        print("Use [SG diffusion] model")
        if "no_prm" in name or "eval" in name:
            return SGDiffusion(dataset_collection)
        else:
            print(f"Apply [permutation] augmentations on {permute_keys}")
            dataset_collection = Permutation(
                dataset_collection,
                permute_keys,
            )
            return SGDiffusion(dataset_collection)

    else:
        raise NotImplementedError()
