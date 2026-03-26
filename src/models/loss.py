from typing import Any, List, Optional, Union, Dict, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn


def make_circumcircle(
  triangles: torch.FloatTensor, 
  normals_unnormed: torch.FloatTensor,
  eps: float=1e-6
) -> torch.FloatTensor:
  edge_2_to_0 = triangles[..., 0, :] - triangles[..., 2, :]
  edge_2_to_1 = triangles[..., 1, :] - triangles[..., 2, :]
  edge_1_to_0 = edge_2_to_0 - edge_2_to_1 
    
  circum_radius = edge_1_to_0.norm(dim=-1, keepdim=True) * edge_2_to_0.norm(dim=-1, keepdim=True) * edge_2_to_1.norm(dim=-1, keepdim=True) / (2.0 * normals_unnormed.norm(dim=-1, keepdim=True) + eps)
    
  circum_center = torch.cross(
    (edge_2_to_0 ** 2).sum(dim=-1, keepdim=True) * edge_2_to_1 - (edge_2_to_1 ** 2).sum(dim=-1, keepdim=True) * edge_2_to_0,
    edge_2_to_0.cross(edge_2_to_1, dim=-1),
    dim=-1
  )
  
  
  circum_center = circum_center / (2.0 * (normals_unnormed ** 2 + eps).sum(dim=-1, keepdim=True))
  
  return circum_radius, circum_center + triangles[..., 2, :]


class ConicalDistanceField(nn.Module):
  def __init__(
    self,
    cone_height: float=.5,
    penalize_outside: bool=True,
    linear_max: int=1000
  ) -> None:
    super().__init__()
    self._cone_height = cone_height
    self._penalize_outside = penalize_outside
    self._linear_max = linear_max
  
  @property
  def cone_height(self) -> float:
    return self._cone_height
  
  @property
  def penalize_outside(self) -> bool:
    return self._penalize_outside
    
  @property
  def linear_max(self) -> int:
    return self._linear_max

  
  def distance_to_conical_axis(
    self,
    points: torch.FloatTensor,
    cone_center: torch.FloatTensor,
    cone_axis: torch.FloatTensor,
    cone_radius: torch.FloatTensor,
    eps: float=1e-6
  ) -> Dict[str, Any]:
    points = points - cone_center[..., None, :]
    length_along_axis = (points * cone_axis[..., None, :]).sum(dim=-1)
    dist_to_axis = (points - length_along_axis[..., None] * cone_axis[..., None, :]).norm(p=2, dim=-1)
    ratio = (1 -  length_along_axis / self._cone_height)
    
    return {
      'dist_to_axis': dist_to_axis / (cone_radius * ratio + eps),
      'length_along_axis': length_along_axis
      }
  
  
  def repulsion_intensity(self, x: torch.FloatTensor, eps: float=1e-6) -> torch.FloatTensor:
    quad_penalty = - (1.0 - 2.0 * self._cone_height) / ((4.0 * self._cone_height ** 2) * x ** 2 + eps) - 1 / (2.0 * self._cone_height) * x +0.25 * (3 - 2 * self._cone_height)
    
    linear_region_mask = (x.le(-self._cone_height) * x.gt(-self._linear_max)).to(dtype=x.dtype)
    
    if self._penalize_outside:
        quad_region_mask = (x.gt(-self._cone_height) * x.lt(self._cone_height)).to(dtype=x.dtype)
    else:
        quad_region_mask = (x.gt(-self._cone_height) * x.lt(0)).to(dtype=x.dtype)

    return (linear_region_mask * (-x + 1 - self._cone_height) +
            quad_region_mask * quad_penalty)
  
  
  def forward(
    self,
    points: torch.FloatTensor,
    cone_center: torch.FloatTensor,
    cone_radius: torch.FloatTensor,
    cone_axis: torch.FloatTensor,
  ) -> torch.FloatTensor:
    
    axis_dist_dict = self.distance_to_conical_axis(points=points, cone_center=cone_center, cone_axis=cone_axis, cone_radius=cone_radius)
    smooth_intensity = self.repulsion_intensity(axis_dist_dict['length_along_axis'])

    
    mask = axis_dist_dict['dist_to_axis'].lt(1).type_as(points).to(points.device)
    return mask * ((1 - axis_dist_dict['dist_to_axis']) * smooth_intensity) ** 2
  
  
class DistanceFieldPenetrationLoss(nn.Module):
  def __init__(
    self,
    cone_height: float=0.5,
    pentalize_outside: bool=True,
    linear_max: int=1000
  ) -> None:
    super(DistanceFieldPenetrationLoss, self).__init__()
    self.cone_height = cone_height
    self.pentalize_outside = pentalize_outside
    
    self.distance_field = ConicalDistanceField(
      cone_height=cone_height,
      penalize_outside=pentalize_outside,
      linear_max=linear_max
    )
    
  def forward(
    self, 
    triangles: torch.FloatTensor,
    collision_inds: torch.IntTensor,
    mesh_ids: torch.IntTensor,
    relation_map: torch.Tensor,
    eps: float=1e-6
  ) -> Dict[str, Any]:
    
    edge_0_to_1 = triangles[..., 1, :] - triangles[..., 0, :]
    edge_0_to_2 = triangles[..., 2, :] - triangles[..., 0, :]
    normals_unnormed = torch.cross(edge_0_to_1, edge_0_to_2, dim=-1)
    circum_raidus, circum_center = make_circumcircle(triangles=triangles, normals_unnormed=normals_unnormed)
    normals = normals_unnormed / ( normals_unnormed.norm(p=2, dim=-1, keepdim=True) + eps)
    circum_center = circum_center.detach()
    circum_raidus = circum_raidus.detach()
    normals = normals.detach()
    
    batch_size = triangles.shape[0]
    valid_inds = collision_inds[..., 0].ge(0).nonzero()
    
    if len(valid_inds) < 1:
      return torch.zeros(batch_size, dtype=triangles.dtype, device=triangles.device)
      
    batch_inds = valid_inds[..., 0]
    faces_inds = valid_inds[..., 1]
    
    recv_faces_inds = collision_inds[batch_inds, faces_inds, 0]
    intr_faces_inds = collision_inds[batch_inds, faces_inds, 1]
    
    recv_mesh_ids = mesh_ids[recv_faces_inds]
    intr_mesh_ids = mesh_ids[intr_faces_inds]

    different_mesh_mask = recv_mesh_ids != intr_mesh_ids
    batch_inds_filtered = batch_inds[different_mesh_mask]
    
    num_collisions = batch_inds_filtered.shape[0]
    
    if num_collisions < 1:
        return torch.zeros(batch_size, dtype=triangles.dtype, device=triangles.device)

    recv_faces_inds_filtered = recv_faces_inds[different_mesh_mask]
    recv_triangles = triangles[batch_inds_filtered, recv_faces_inds_filtered]
    recv_normals = normals[batch_inds_filtered, recv_faces_inds_filtered]
    recv_circum_radius = circum_raidus[batch_inds_filtered, recv_faces_inds_filtered]
    recv_circum_center = circum_center[batch_inds_filtered, recv_faces_inds_filtered]
    
    intr_faces_inds_filtered = intr_faces_inds[different_mesh_mask]
    intr_triangles = triangles[batch_inds_filtered, intr_faces_inds_filtered]
    intr_normals = normals[batch_inds_filtered, intr_faces_inds_filtered]
    intr_circum_radius = circum_raidus[batch_inds_filtered, intr_faces_inds_filtered]
    intr_circum_center = circum_center[batch_inds_filtered, intr_faces_inds_filtered]

    recv_mesh_ids_filtered = recv_mesh_ids[different_mesh_mask]
    intr_mesh_ids_filtered = intr_mesh_ids[different_mesh_mask]

    recv_is_supporter = relation_map[recv_mesh_ids_filtered, intr_mesh_ids_filtered] == 1
    intr_is_supporter = relation_map[intr_mesh_ids_filtered, recv_mesh_ids_filtered] == 1

    recv_triangles_for_df = recv_triangles.clone()
    intr_triangles_for_df = intr_triangles.clone()

    recv_triangles_for_df[recv_is_supporter] = recv_triangles[recv_is_supporter].detach()
    
    intr_triangles_for_df[intr_is_supporter] = intr_triangles[intr_is_supporter].detach()
    
    recv_dist_field = self.distance_field.forward(
      intr_triangles_for_df,
      recv_circum_center,
      recv_circum_radius,
      recv_normals
    )
    
    intr_dist_field = self.distance_field.forward(
      recv_triangles_for_df,
      intr_circum_center,
      intr_circum_radius,
      intr_normals
    )
    
    recv_loss = (-1 * recv_dist_field[..., None] * intr_normals[..., None, :]).norm(dim=-1).pow(2).mean()
    intr_loss = (-1 * intr_dist_field[..., None] * recv_normals[..., None, :]).norm(dim=-1).pow(2).mean()
    loss = recv_loss + intr_loss
    return loss


