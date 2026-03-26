import os
import argparse
import time
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusers.training_utils import EMAModel

from src.utils import *
from src.data import get_encoded_dataset, filter_function
from src.models import model_from_config, optimizer_from_config

import trimesh
from trimesh.visual import ColorVisuals

from transformers import AutoModel

def box2transform(box, scene_id, obj_id):
    T = box["translations"][scene_id][obj_id]
    size = box["sizes"][scene_id][obj_id]
    R = box["angles"][scene_id][obj_id]
    
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T
    
    return transform, {"R": R, "T": T, "size": size}
    
def main():
    parser = argparse.ArgumentParser(
        description="Train a generative model on scene bounding boxes, conditioned on scene graphs"
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag that refers to the current experiment"
    )
    parser.add_argument(
        "--fvqvae_tag",
        type=str,
        help="Tag that refers to the fVQ-VAE experiment"
    )
    parser.add_argument(
        "--fvqvae_epoch",
        type=int,
        default=1999,
        help="Epoch of the pretrained fVQ-VAE"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="The epoch to load the checkpoint from"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="The number of processed spawned by the batch provider (default=0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the PRNG (default=0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device to use for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="weight of guidance"
    )
    

    args = parser.parse_args()

    if args.seed is not None and args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"You have chosen to seed([{args.seed}]) the experiment")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
    print(f"Run code on device [{device}]\n")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.tag is None:
        tag = time.strftime("%Y-%m-%d_%H:%M") + "_" + \
            os.path.split(args.config_file)[-1].split()[0]
    else:
        tag = args.tag

    model_folder = "./dataset/meshes"

    exp_dir = os.path.join(args.output_dir, tag)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    save_experiment_params(args, tag, exp_dir)

    config: Dict[str, Dict[str, Any]] = load_config(args.config_file)

    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"]),
    )
    np.savez(
        os.path.join(exp_dir, "bounds.npz"),
        translations=train_dataset.bounds["translations"],
        sizes=train_dataset.bounds["sizes"],
        angles=train_dataset.bounds["angles"]
    )
    print(f"Training set has bounds: {train_dataset.bounds}")
    print(f"Load [{len(train_dataset)}] training scenes\n")

    config["data"]["encoding_type"] += "_eval"
    val_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=os.path.join(exp_dir, "bounds.npz"),
        augmentations=None,
        split=config["validation"].get("splits", ["test"]),
    )
    print(f"Load [{len(val_dataset)}] validation scenes\n")

    print("Creating data loader\n")
    train_loader = MultiEpochsDataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=train_dataset.collate_fn,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["validation"]["batch_size"],
        num_workers=args.n_workers,
        pin_memory=False,
        collate_fn=val_dataset.collate_fn,
        shuffle=False
    )

    kwargs = {}

    model = model_from_config(
        config["model"],
        0,
        train_dataset.n_predicate_types,
        train_dataset.n_region_types,
        **kwargs
    ).to(device)
    optimizer = optimizer_from_config(
        config["training"]["optimizer"],
        filter(lambda p: p.requires_grad, model.parameters())
    )
    
    print('\nLoading DINO model...')
    dinov2_model_name="facebook/dinov2-base"
    device="cuda"
    dinov2_model = AutoModel.from_pretrained(dinov2_model_name).to(device)
    dinov2_model.eval()
    print("Model loaded!\n")

    save_model_architecture(model, exp_dir)

    ema_config = config["training"]["ema"]
    if ema_config["use_ema"]:
        print(f"Use exponential moving average (EMA) for model parameters\n")
        ema_states = EMAModel(
            model.parameters(),
            decay=ema_config["max_decay"],
            min_decay=ema_config["min_decay"],
            update_after_step=ema_config["update_after_step"],
            use_ema_warmup=ema_config["use_warmup"],
            inv_gamma=ema_config["inv_gamma"],
            power=ema_config["power"]
        )
        ema_states.to(device)
    else:
        ema_states: EMAModel = None

    try:
        start_epoch = 0
        start_epoch = load_checkpoints(model, ckpt_dir, ema_states, optimizer, args.checkpoint_epoch, device)
    except:
        print("Compiling model for performance...")
        model = torch.compile(model, mode="max-autotune")
        print("Model compiled successfully.")
        
        start_epoch = 0
        start_epoch = load_checkpoints(model, ckpt_dir, ema_states, optimizer, args.checkpoint_epoch, device)

    print("\n================ Testing ================")
    if ema_states is not None:
        ema_states.store(model.parameters())
        ema_states.copy_to(model.parameters())
    model.eval()
    
    num_timesteps = 1000
    lr = args.lr 

    scene_metadata = []

    with torch.no_grad():
        for val_b, val_batch in enumerate(val_loader):
            
            for k, v in val_batch.items():
                if not isinstance(v, list) and k not in ['rgb']:
                    val_batch[k] = v.to(device)
                    
            if model.use_image:
                outputs = dinov2_model(pixel_values=val_batch['image'])
                patch_tokens = outputs.last_hidden_state[:, 1:, :]
                val_batch['dinov2_feature'] = patch_tokens
            else:
                val_batch['dinov2_feature'] = None
                
            
            if True:
                _boxes_pred = model.generate_samples(val_batch, dataset=val_dataset, num_timesteps=num_timesteps, lr=lr, scene_name=val_b, seed=args.seed, bounds=train_dataset.bounds)
                
                boxes_pred = _boxes_pred.cpu().numpy()
                bbox_params = {
                    "translations": boxes_pred[..., :3],
                    "sizes": boxes_pred[..., 3:6],
                    "angles": boxes_pred[..., 6:]
                }
                boxes_pred = val_dataset.post_process(bbox_params)
                
                val_batch['scene_uids'] = [os.path.dirname(path) for path in val_batch['scene_uids']]
                
                vis_folder = os.path.join(f"{exp_dir}_val", f'vis_{num_timesteps}')
                os.makedirs(vis_folder, exist_ok=True)
                
                scene_models_folder = os.path.join(vis_folder, f'scene_models_{lr}')
                os.makedirs(scene_models_folder, exist_ok=True)
                
                for scene_id, scene_folder in enumerate(val_batch['scene_uids']):
                    relative_path = os.path.relpath(scene_folder, config["data"]["dataset_directory"])
                    
                    trimesh_scene = trimesh.Scene()
                    
                    all_min = np.inf * np.ones(3)
                    all_max = -np.inf * np.ones(3)
                    
                    centroids = []
                    scene_boxes = []
                    
                    for obj_id, name in enumerate(val_batch['obj_names'][scene_id][1:]):
                        name = str(name)
                        transform, box = box2transform(boxes_pred, scene_id, obj_id)
                
                        box = {k:v.tolist() for k,v in box.items()}
                        box["obj_name"] = name
                        scene_boxes.append(box)
                        
                        model_path = os.path.join(model_folder, f"{name}_unpacked.glb")
                        mesh = trimesh.load(model_path, force='mesh')
                        mesh.apply_translation(-val_batch['bbox_center'][scene_id][obj_id+1])
                        mesh.apply_transform(transform)
                        centroids.append(mesh.centroid)
                        
                        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                            mesh.visual = ColorVisuals(mesh, vertex_colors=np.ones((mesh.vertices.shape[0], 4), dtype=np.uint8) * 255)
                        
                        trimesh_scene.add_geometry(mesh, node_name=f"{name}_{obj_id}")

                        vertices_min = mesh.vertices.min(axis=0)
                        vertices_max = mesh.vertices.max(axis=0)
                        
                        all_min = np.minimum(all_min, vertices_min)
                        all_max = np.maximum(all_max, vertices_max)
                        
                    object_center = np.mean(centroids, axis=0)
                    floor_size = np.max(all_max - all_min) * 1.2
                    floor = trimesh.creation.box(extents=[floor_size, floor_size, 0.001])  
                    rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
                    floor.apply_transform(rotation_matrix)
                    floor.apply_translation([object_center[0],0, object_center[2]])  

                    trimesh_scene.add_geometry(floor, node_name="floor")
                    
                    scene_index = val_b * val_loader.batch_size + scene_id
                    
                    scene_filename = f"epoch_{start_epoch}_scene_{scene_index}_{relative_path}.glb"
                    scene_path = os.path.join(scene_models_folder, scene_filename)
                    trimesh_scene.export(scene_path)
                    
                    scene_metadata.append({
                        'id': scene_index,
                        'name': relative_path,
                        'model_path': os.path.relpath(scene_path, vis_folder),
                        'boxes': scene_boxes
                    })
                
                metadata_path = os.path.join(vis_folder, f"epoch_{start_epoch}_scenes_metadata_{lr}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(scene_metadata, f, indent=2)

        print("================ Testing ================\n")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
