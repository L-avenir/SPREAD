from typing import *
from torch import Tensor, LongTensor

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from tqdm import tqdm
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from .networks import *

import numpy as np

from .perceiver_pytorch import Perceiver

from mesh_intersection.bvh_search_tree import BVH
from src.models.loss import DistanceFieldPenetrationLoss

import os, trimesh
from scipy.spatial import ConvexHull

def points_to_convex_hull_distance(points: torch.Tensor, hull: torch.Tensor) -> torch.Tensor:
    if hull.shape[0] < 3:
        if hull.shape[0] > 0:
            hull_center = hull.mean(0)
            return torch.linalg.norm(points - hull_center, dim=1)
        return torch.full((points.shape[0],), float('inf'), device=points.device)

    hull_p1 = hull
    hull_p2 = torch.roll(hull, shifts=-1, dims=0)
    
    v = hull_p2 - hull_p1
    w = points.unsqueeze(1) - hull_p1.unsqueeze(0)
    
    cross_z = v[:, 0] * w[..., 1] - v[:, 1] * w[..., 0]
    
    is_inside = torch.all(cross_z >= -1e-6, dim=1)
    
    dot_wv = torch.einsum('phd,hd->ph', w, v)
    dot_vv = torch.einsum('hd,hd->h', v, v)
    t = dot_wv / (dot_vv.unsqueeze(0) + 1e-8)
    t = torch.clamp(t, 0, 1)
    
    projection = hull_p1.unsqueeze(0) + t.unsqueeze(2) * v.unsqueeze(0)
    
    dist_to_edges = torch.linalg.norm(points.unsqueeze(1) - projection, dim=2)
    
    min_dist_to_hull, _ = torch.min(dist_to_edges, dim=1)
    
    final_distances = torch.where(is_inside, 0.0, min_dist_to_hull)
    
    return final_distances

def torch_convex_hull_2d_nongrad_efficient(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] <= 3:
        return points

    with torch.no_grad():
        numpy_points = points.cpu().numpy()

        hull = ConvexHull(numpy_points)

        hull_indices = torch.from_numpy(hull.vertices).long()

        hull_points = points[hull_indices]

    return hull_points, hull_indices

def compute_loss_relation_from_points(
    xz_coords: List[torch.Tensor],
    relation_mat: torch.Tensor,
    xz_projections: List[torch.Tensor]
) -> torch.Tensor:
    all_pair_losses = []
    s_indices, o_indices = torch.where(relation_mat == 1)
    device = relation_mat.device
    
    for s_idx, o_idx in zip(s_indices, o_indices):
        supporter_hull = xz_projections[s_idx].detach()
        
        object_points = xz_coords[o_idx]

        if object_points.shape[0] == 0 or supporter_hull.shape[0] < 3:
            continue

        distances = points_to_convex_hull_distance(object_points, supporter_hull)
        
        num_outside_points = torch.sum(distances > 1e-6).float()
        
        if num_outside_points > 0:
            sum_of_outside_distances = torch.sum(distances)
            
            mean_outside_distance = sum_of_outside_distances / num_outside_points
            all_pair_losses.append(mean_outside_distance)

    if not all_pair_losses:
        return torch.tensor(0.0, requires_grad=True, device=device)
    
    total_loss = torch.mean(torch.stack(all_pair_losses))
    return total_loss

def _get_distance_ray_casting(
    ray_origins: torch.Tensor,
    triangles: torch.Tensor,
    mesh_ids: torch.Tensor,
    device: torch.device = torch.device('cuda')
) -> torch.Tensor:
    num_rays = ray_origins.shape[0]
    ray_directions = torch.tensor([[0., -1., 0.]], device=device).expand(num_rays, -1)

    num_faces = triangles.shape[0]
    v0 = triangles[:, 0, :].unsqueeze(0)
    edge1 = triangles[:, 1, :] - v0
    edge2 = triangles[:, 2, :] - v0

    ray_origins_exp = ray_origins.unsqueeze(1).expand(-1, num_faces, -1)
    ray_directions_exp = ray_directions.unsqueeze(1).expand(-1, num_faces, -1)

    epsilon = 1e-6
    pvec = torch.cross(ray_directions_exp, edge2, dim=-1)
    det = torch.sum(edge1 * pvec, dim=-1)

    mask_parallel = torch.abs(det) < epsilon
    inv_det = torch.zeros_like(det)
    inv_det[~mask_parallel] = 1.0 / det[~mask_parallel]

    tvec = ray_origins_exp - v0
    u = torch.sum(tvec * pvec, dim=-1) * inv_det

    qvec = torch.cross(tvec, edge1, dim=-1)
    v = torch.sum(ray_directions_exp * qvec, dim=-1) * inv_det
    
    t = torch.sum(edge2 * qvec, dim=-1) * inv_det

    ray_indices = torch.arange(num_rays, device=device).unsqueeze(1)
    face_mesh_ids = mesh_ids.unsqueeze(0)
    self_hit_mask = (ray_indices == face_mesh_ids)
    valid_hit_mask = (~self_hit_mask) & (~mask_parallel) & (u >= -epsilon) & (v >= -epsilon) & ((u + v) <= 1.0 + epsilon) & (t > epsilon)

    t[~valid_hit_mask] = float('inf')

    min_distances, _ = torch.min(t, dim=1)

    fallback_mask = torch.isinf(min_distances)
    fallback_distances = ray_origins[fallback_mask, 1]
    min_distances[fallback_mask] = fallback_distances
    
    return min_distances

def descale(x, minimum, maximum):
    x = (x + 1) / 2
    x = x * (maximum - minimum) + minimum
    return x
    
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    proj = (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2 - proj, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack((b1, b2, b3), dim=-1)
    return R

def boxes_pred_to_transforms(boxes_pred: torch.Tensor,
                             bounds: dict):
    B, N, _ = boxes_pred.shape
    device, dtype = boxes_pred.device, boxes_pred.dtype

    translations_6d = boxes_pred[..., :3]
    sizes_6d        = boxes_pred[..., 3:6]
    rot6d          = boxes_pred[..., 6:]

    t_min, t_max = bounds['translations']
    s_min, s_max = bounds['sizes']
    T = descale(translations_6d, t_min, t_max)
    S = descale(sizes_6d,        s_min, s_max)

    R = rotation_6d_to_matrix(rot6d)

    RT = torch.cat([R, T.unsqueeze(-1)], dim=-1)

    bottom = torch.tensor([0, 0, 0, 1],
                          device=device, dtype=dtype) \
                 .view(1, 1, 1, 4) \
                 .expand(B, N, 1, 4)

    transforms = torch.cat([RT, bottom], dim=2)

    metadata = {
        'R':    R,
        'T':    T,
        'size': S
    }

    return transforms, metadata

def build_triangles_and_mesh_ids(
     mesh_list: List[trimesh.Trimesh],
     translations_np: np.ndarray,
     RT: torch.Tensor,
     convex_hull_inds: List[torch.LongTensor] = None,
     device: torch.device = torch.device('cuda')
 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.LongTensor]]:
    if translations_np.shape[0] > len(mesh_list):
        translations_np = translations_np[1:, ...]
        RT = RT[:, 1:, ...]
        
    translations = torch.from_numpy(translations_np).float().to(device)
    translations = translations.unsqueeze(0)
    B, N, _ = translations.shape

    RT = RT.to(device)
    assert RT.shape == (B, N, 4, 4), f"RT must be (1,{N},4,4), got {RT.shape}"

    triangles_per_batch = []
    mesh_id_list = []
    obj_heights_list = []
    point_A_list = []
    xz_projections_list = []
    all_xz_coords_list = []
    all_hull_inds_list = [] if convex_hull_inds is None else convex_hull_inds

    for i, mesh in enumerate(mesh_list):
        verts = torch.from_numpy(mesh.vertices).float().to(device)
        faces = torch.from_numpy(mesh.faces).long().to(device)
        
        V = verts.shape[0]
        F = faces.shape[0]

        t_i = translations[:, i, :]
        RT_i = RT[:, i, :, :]
        v = verts.unsqueeze(0).expand(B, -1, -1)
        v_trans = v + t_i.unsqueeze(1)
        ones = torch.ones(B, V, 1, device=device)
        v_hom = torch.cat([v_trans, ones], dim=2)
        v_rt = torch.matmul(v_hom, RT_i.transpose(1, 2))
        v_rt3 = v_rt[..., :3].squeeze(0)
        
        mesh_triangles = v_rt3[faces]

        min_height = torch.min(v_rt3[:, 1])

        v_min, _ = torch.min(v_rt3, dim=0)
        v_max, _ = torch.max(v_rt3, dim=0)
        
        geom_center = (v_max + v_min) / 2.0
        obj_height = geom_center[1] - min_height
        obj_heights_list.append(obj_height)
        point_A_list.append(geom_center)
        
        xz_coords = v_rt3[:, [0, 2]]
        
        all_xz_coords_list.append(xz_coords)

        if convex_hull_inds is None:
            hull_vertices_tensor, hull_inds = torch_convex_hull_2d_nongrad_efficient(xz_coords)
            xz_projections_list.append(hull_vertices_tensor)
            all_hull_inds_list.append(hull_inds)
        else:
            hull_vertices_tensor = xz_coords[convex_hull_inds[i]]
            xz_projections_list.append(hull_vertices_tensor)
        
        tris = mesh_triangles.unsqueeze(0)
        triangles_per_batch.append(tris)

        ids = torch.full((F,), i, dtype=torch.long, device=device)
        mesh_id_list.append(ids)
    
    triangles = torch.cat(triangles_per_batch, dim=1)
    mesh_ids = torch.cat(mesh_id_list, dim=0)
    obj_heights = torch.stack(obj_heights_list)
    all_points_A = torch.stack(point_A_list)

    distances = _get_distance_ray_casting(all_points_A, triangles.squeeze(0).detach(), mesh_ids, device)
    
    return triangles, mesh_ids, distances, obj_heights, xz_projections_list, all_xz_coords_list, all_hull_inds_list

def batch_box2transform(box):
    T = box["translations"][:,1:,:]
    S = box["sizes"][:,1:,:]
    R = box["angles"][:,1:,:,:]
    B, N, _ = T.shape

    transform = np.tile(np.eye(4), (B, N, 1, 1))

    transform[:, :, :3, :3] = R

    transform[:, :, :3, 3] = T

    return transform

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * 
        (-math.log(10000.0) / d_model)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


class Sg2ScCASTDiffusion_fixed_michelangelo(nn.Module):
    def __init__(self,
        num_regions: int, num_preds: int, config: Dict[str, Any],
        diffusion_type="ddpm",
        cfg_drop_ratio=0.2,
        use_objfeat=True
    ):
        super().__init__()
        
        self.config = config
        self.use_graph = config.get("use_graph", True)
        self.use_image = config.get("use_image", True)
        self.use_point = config.get("use_point", True)
        self.beta_schedule = config.get("beta_schedule", "linear")
        print(f"Use graph: {self.use_graph}, use image: {self.use_image}, use point: {self.use_point}, beta_schedule: {self.beta_schedule}")

        if diffusion_type == "ddpm":
            self.scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule=self.beta_schedule,
                variance_type="fixed_small",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
            self.scheduler_train = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                variance_type="fixed_small",
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        elif diffusion_type == "ddim":
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.0001, beta_end=0.02,
                beta_schedule=self.beta_schedule,
                prediction_type="epsilon",
                clip_sample=True,
                clip_sample_range=1.
            )
        else:
            raise NotImplementedError

        self.network = Sg2ScTransformerDiffusionWrapper(
            region_dim=num_regions+1,
            edge_dim=num_preds+1,
            t_dim=128, attn_dim=512,
            global_condition_dim=None,
            context_dim=768,
            n_heads=8, n_layers=6,
            gated_ff=True, dropout=0.1, ada_norm=True,
            cfg_drop_ratio=cfg_drop_ratio,
            use_objfeat=use_objfeat,
        )

        self.num_regions = num_regions
        self.num_preds = num_preds
        self.use_objfeat = use_objfeat

        self.cfg_scale = 1.
        
        self.search_tree = BVH(max_collisions=64)
        self.loss_fn = DistanceFieldPenetrationLoss(cone_height=0.5).to('cuda')
        
        self.bounds = {
            "translations": (
                torch.tensor(
                    [-1.07434147e+01, -4.46544331e-03, -8.64576626e+00],
                    dtype=torch.float32,
                    device='cuda'
                ),
                torch.tensor(
                    [10.83410962,  3.08334312, 10.28521920],
                    dtype=torch.float32,
                    device='cuda'
                )
            ),
            "sizes": (
                torch.tensor(
                    [0.0, 0.0, 0.0],
                    dtype=torch.float32,
                    device='cuda'
                ),
                torch.tensor(
                    [1.0, 1.0, 1.0],
                    dtype=torch.float32,
                    device='cuda'
                )
            ),
        }
        

    def compute_pc_features_chamfer_distance(self, batch_points, batch_normals, bbox, dataset):
        bbox_params = {
                        "translations": bbox[..., :3].cpu().numpy(),
                        "sizes": bbox[..., 3:6].cpu().numpy(),
                        "angles": bbox[..., 6:].cpu().numpy()
                        }
        boxes_pred = dataset.post_process(bbox_params)
        local_to_world_transforms = batch_box2transform(boxes_pred)
        local_to_world_transforms = torch.from_numpy(local_to_world_transforms).float().to(batch_points.device)
        
        B, N, _ = bbox.shape
        N_obj = N - 1
        _, _, num_points, _ = batch_points.shape
        
        points = batch_points[:, 1:].float()
        normals = batch_normals[:, 1:].float()
        points_homogeneous = F.pad(points, (0, 1), 'constant', 1.0)

        points_in_world_homo = torch.matmul(
            local_to_world_transforms, points_homogeneous.permute(0, 1, 3, 2)
        )
        points_in_world = points_in_world_homo.permute(0, 1, 3, 2)[..., :3]

        R = local_to_world_transforms[..., :3, :3]
        normals_in_world = torch.matmul(R.unsqueeze(2), normals.unsqueeze(-1)).squeeze(-1)
        
        all_dists = torch.zeros((B, N_obj, num_points), device=points.device)
        all_signs = torch.zeros((B, N_obj, num_points), device=points.device)

        for i in range(N_obj):
            src = points_in_world[:, i]
            others = [points_in_world[:, j] for j in range(N_obj) if j != i]
            tgt = torch.cat(others, dim=1)

            d = torch.cdist(src, tgt, p=2)
            min_dists, idx_src = torch.min(d, dim=2)
            all_dists[:, i] = min_dists

            norms_other = torch.cat([normals_in_world[:, j] for j in range(N_obj) if j != i], dim=1)
            sel_norms = torch.gather(norms_other, 1, idx_src.unsqueeze(-1).expand(-1, -1, 3))
            nearest_pts = torch.gather(tgt, 1, idx_src.unsqueeze(-1).expand(-1, -1, 3))

            dir_vec = points_in_world[:, i] - nearest_pts
            signs = torch.sign((dir_vec * sel_norms).sum(dim=-1))
            all_signs[:, i] = signs

        signed_dists = all_dists * all_signs

        pc_feature = torch.cat([points, signed_dists.unsqueeze(-1)], dim=-1)
        
        pc_feature_padded = torch.zeros((B, N, num_points, 4), device=pc_feature.device)
        pc_feature_padded[:, 1:] = pc_feature
        
        return pc_feature_padded

    @torch.no_grad()
    def generate_samples(self,
        sample_params: Dict[str, Tensor],
        dataset = None,
        vqvae_model: nn.Module = None,
        num_timesteps: Optional[int]=100,
        lr: float=1e-5,
        scene_name: int=0,
        seed = 0,
        bounds = None,
        cfg_scale=1.
    ):
        self.assets_folder = "./dataset/meshes"
        miche_feat = sample_params["miche_feature"]
        dinov2_feat = sample_params["dinov2_feature"]
        e = sample_params["edges"]
        r = sample_params["regions"]
        mask = sample_params["obj_masks"]
        boxes = sample_params["boxes"]
        mesh_names = sample_params['obj_names'][0]
        
        relation_mat = sample_params['edges'][0]
        relation_mat = relation_mat[1:, 1:]
        
        optmized_hull = None
        
        alpha_train = self.scheduler_train.alphas_cumprod.cpu().numpy()
        alpha_new = self.scheduler.alphas_cumprod.cpu().numpy()
        mapping = []
        
        for a_new in alpha_new:
            idx = np.argmin(np.abs(alpha_train - a_new))
            mapping.append(int(idx))
        mapping = torch.tensor(mapping, dtype=torch.int64, device=boxes.device)
        meshes = []
        for id, name in enumerate(mesh_names):
            mesh_path = os.path.join(self.assets_folder, f"{name}_unpacked.glb")
            if os.path.exists(mesh_path):
                mesh = trimesh.load(os.path.join(self.assets_folder, f"{name}_unpacked.glb"), force='mesh')
                meshes.append(mesh)
        
        if not self.use_image:
            dinov2_feat = torch.zeros((1,256,768)).to(e.device)
        if not self.use_graph:
            e = torch.zeros_like(e)
            r = torch.zeros_like(r)
        
        self.cfg_scale = cfg_scale
        B, N, device = miche_feat.shape[0], miche_feat.shape[1], miche_feat.device

        boxes = torch.randn(B, N, 12).to(device)

        box_mask = mask.unsqueeze(-1)
        boxes = boxes * box_mask

        if num_timesteps is None:
            num_timesteps = self.scheduler.config.num_train_timesteps
        num_timesteps = 1000
        self.scheduler.set_timesteps(num_timesteps)
        step_count = -1
        for t in tqdm(self.scheduler.timesteps, desc=f"Generating scenes {scene_name}", ncols=125):
            step_count += 1
            
            points = sample_params['xyz'].repeat(B, 1, 1, 1) 
            normals = sample_params['normal'].repeat(B, 1, 1, 1)
            pc_features = self.compute_pc_features_chamfer_distance(points, normals, boxes, dataset)
            
            train_t = mapping[t] if self.scheduler_train is not None else t
            
            
            pred = self.network(boxes, pc_features, miche_feat, e, r, train_t, condition=dinov2_feat, mask=mask, cfg_scale=cfg_scale) * box_mask
                        
            boxes = self.scheduler.step(pred, t, boxes).prev_sample * box_mask
            
        for j in tqdm(range(100), desc="Generating scenes", ncols=125):
            self.scheduler.set_timesteps(1)
            for t in self.scheduler.timesteps:
                step_count += 1
                
                points = sample_params['xyz'].repeat(B, 1, 1, 1) 
                normals = sample_params['normal'].repeat(B, 1, 1, 1)
                pc_features = self.compute_pc_features_chamfer_distance(points, normals, boxes, dataset)
                
                pred = self.network(boxes, pc_features, miche_feat, e, r, t, condition=dinov2_feat, mask=mask, cfg_scale=cfg_scale) * box_mask
                boxes = self.scheduler.step(pred, t, boxes).prev_sample * box_mask
                
                with torch.enable_grad():
                    graded_boxes = boxes.clone().detach().requires_grad_(True)
                    
                    transforms, _ = boxes_pred_to_transforms(graded_boxes, self.bounds)
                    triangles, mesh_ids, ray_distances, obj_heights, xz_projections, xz_verts, optmized_hull = build_triangles_and_mesh_ids(meshes, -sample_params['bbox_center'][0], transforms, optmized_hull)
                    
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                    lambda_colli = 15
                    lambda_relation = 2e0
                    lambda_gravity = 2e0
                    
                    grad = torch.zeros_like(graded_boxes)
                    collision_inds = self.search_tree(triangles)
                    loss_colli = self.loss_fn(triangles, collision_inds, mesh_ids, relation_mat)
                    if loss_colli == 0:
                        grad_colli = torch.zeros_like(graded_boxes)
                    else:
                        grad_colli = torch.autograd.grad(loss_colli, graded_boxes, allow_unused=True, retain_graph=True)[0]
                    if grad_colli is not None:
                        grad_colli[..., 1] = torch.where(grad_colli[..., 1] < 0, grad_colli[..., 1] * 3, grad_colli[..., 1])
                        grad = grad + grad_colli * lambda_colli
                    
                    loss_relation = compute_loss_relation_from_points(xz_verts, relation_mat, xz_projections)
                    grad_relation = torch.autograd.grad(loss_relation, graded_boxes, allow_unused=True, retain_graph=True)[0]
                    if grad_relation is not None:
                        grad = grad + grad_relation * lambda_relation
                        
                    if j >= 50:
                        res_dis = ray_distances - obj_heights - 5e-4
                        loss_gravity = torch.sum(torch.abs(res_dis[(res_dis > 2e-3) | (res_dis < 0)])).requires_grad_()
                        grad_gravity = torch.autograd.grad(loss_gravity, graded_boxes, allow_unused=True, retain_graph=True)[0]
                        if grad_gravity is not None:
                            grad_gravity[..., 0] = 0
                            grad_gravity[..., 2:] = 0
                            grad = grad + grad_gravity * lambda_gravity
                        
                    if grad is None:
                        grad = torch.zeros_like(graded_boxes)
                    
                boxes = boxes - lr * grad
                
        boxes = boxes[:, 1:, :]

        return boxes


class Sg2ScTransformerDiffusionWrapper(nn.Module):
    def __init__(self,
        region_dim: int, edge_dim: int,
        attn_dim=512, t_dim=128,
        global_dim: Optional[int]=None,
        global_condition_dim: Optional[int]=None,
        context_dim: Optional[int]=None,
        n_heads=8, n_layers=5,
        gated_ff=True, dropout=0.1, ada_norm=True,
        cfg_drop_ratio=0.2,
        use_objfeat=True
    ):
        super().__init__()

        if not ada_norm:
            global_dim = t_dim
            
        self.pe_embedding = sinusoidal_positional_encoding(max_len = 100, d_model = 128)

        self.miche_proj = nn.Linear(64, attn_dim)
        self.box_proj = nn.Linear(3 + 6 + 3 + 128, attn_dim)
        
        self.pc_preceiver = Perceiver(
                                input_channels = 4,
                                input_axis = 2,
                                num_freq_bands = 64,
                                max_freq = 1120.,
                                depth = 4,
                                num_latents = 256,
                                latent_dim = 512,
                                cross_heads = 1,
                                latent_heads = 8,
                                cross_dim_head = 64,
                                latent_dim_head = 64,
                                num_classes = 1000,
                                attn_dropout = 0.,
                                ff_dropout = 0.,
                                weight_tie_layers = False,
                                fourier_encode_data = True,
                                self_per_cross_attn = 2,
                            )

        self.edge_embed = nn.Sequential(
            nn.Embedding(edge_dim, attn_dim//4),
            nn.GELU(),
            nn.Linear(attn_dim//4, attn_dim//4),
        )
        self.region_embed = nn.Sequential(
            nn.Embedding(region_dim, attn_dim//4),
            nn.GELU(),
            nn.Linear(attn_dim//4, attn_dim//4),
        )
        self.edge_proj_in = nn.Linear(attn_dim //2 , attn_dim // 4)
        
        self.time_embed = nn.Sequential(
            Timestep(t_dim),
            TimestepEmbed(t_dim, t_dim)
        )

        if global_condition_dim is not None:
            self.global_condition_embed = nn.Sequential(
                nn.Linear(global_condition_dim, t_dim),
                nn.GELU(),
                nn.Linear(t_dim, t_dim)
            )

        print("Using Fixed Miche and Point Cloud in Sg2Sc Transformer Diffusion Wrapper")
        self.transformer_blocks = nn.ModuleList([
            SimpleMultiModalGraphTransformerBlock(
                attn_dim, attn_dim//4, attn_dim, global_dim,
                context_dim, t_dim,
                n_heads, gated_ff, dropout, ada_norm, use_pc=True, use_fixed_miche=True
            ) for _ in range(n_layers)
        ])

        self.proj_out = nn.Sequential(
            nn.LayerNorm(attn_dim),
            nn.Linear(attn_dim, 12)
        )

        self.region_dim = region_dim
        self.attn_dim = attn_dim
        self.edge_dim = edge_dim
        self.use_global_info = global_dim is not None
        self.cfg_drop_ratio = cfg_drop_ratio

    def forward(self,
        box: Tensor, pc_feat: Tensor, miche_feat: Tensor, e: Tensor, r: Optional[Tensor],
        t: LongTensor, global_condition: Optional[Tensor]=None,
        condition: Optional[Tensor]=None,
        mask: Optional[LongTensor]=None, condition_mask: Optional[LongTensor]=None,
        cfg_scale=1.
    ):
        with record_function("WRAPPER::PREP_AND_EMBEDDINGS"):
            if not torch.is_tensor(t):
                if isinstance(t, (int, float)):
                    t = torch.tensor([t], device=pc_feat.device)
                else:
                    assert len(t) == pc_feat.shape[0]
                    t = torch.tensor(t, device=pc_feat.device)
            else:
                if t.dim() == 0:
                    t = t.unsqueeze(-1).to(pc_feat.device)
            t = t * torch.ones(pc_feat.shape[0], dtype=t.dtype, device=t.device)
            
            B, N, _, _ = pc_feat.shape

            enc_pc = self.pe_embedding[:box.shape[1], :].unsqueeze(0).to(box.device)
            enc_pc = enc_pc.expand(B, N, -1)
            box = torch.cat([box, enc_pc], dim=-1)
        
            box_emb = self.box_proj(box).view(B*N,1,self.attn_dim)
            miche_emb = self.miche_proj(miche_feat).view(B*N,256,self.attn_dim)
        
        pc_emb = self.pc_preceiver(pc_feat, return_embeddings=True)
        
        with record_function("WRAPPER::GRAPH_AND_TIME_EMBEDDINGS"):
            e_emb = self.edge_embed(e)
            r_emb = self.region_embed(r)
            e_emb = self.edge_proj_in(torch.cat([e_emb, r_emb], dim=-1))
            
            t_emb = self.time_embed(t)
            if self.use_global_info:
                y_emb = t_emb
            else:
                y_emb =None
            if global_condition is not None:
                t_emb += self.global_condition_embed(global_condition)

            if self.training and self.cfg_drop_ratio > 0.:
                assert cfg_scale == 1., "Do not use `cfg_scale` during training"
                empty_e_emb = torch.zeros_like(e_emb[0]).unsqueeze(0)
                empty_prob = torch.rand(e_emb.shape[0], device=e_emb.device) < self.cfg_drop_ratio
                
                empty_e_emb_broadcasted = empty_e_emb.expand_as(e_emb)
                e_emb = torch.where(empty_prob.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), empty_e_emb_broadcasted, e_emb)

            if not self.training and cfg_scale != 1.:
                empty_e_emb = torch.zeros_like(e_emb)
                e_emb = torch.cat([empty_e_emb, e_emb], dim=0)
                x_emb = torch.cat([x_emb, x_emb], dim=0)
                y_emb = torch.cat([y_emb, y_emb], dim=0) if y_emb is not None else None
                t_emb = torch.cat([t_emb, t_emb], dim=0)
                if condition is not None:
                    condition = torch.cat([condition, condition], dim=0)
                if mask is not None:
                    mask = torch.cat([mask, mask], dim=0)
                if condition_mask is not None:
                    condition_mask = torch.cat([condition_mask, condition_mask], dim=0)

        for block in self.transformer_blocks:
            box_emb, miche_emb, pc_emb, e_emb, y_emb = block(box_emb, miche_emb, pc_emb, e_emb, y_emb, t_emb, condition, mask, condition_mask, B)

        with record_function("WRAPPER::FINAL_PROJECTION"):
            out_box = self.proj_out(box_emb.view(B,N,self.attn_dim))
            if mask is not None:
                out_box = out_box * mask.unsqueeze(-1)

            if not self.training and cfg_scale != 1.:
                out_box_uncond, out_box_cond = out_box.chunk(2, dim=0)
                out_box = out_box_uncond + cfg_scale * (out_box_cond - out_box_uncond)

        return out_box
