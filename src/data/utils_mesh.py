import open3d as o3d
import numpy as np
import torch

class Mesh:
    def __init__(self, vertices, faces):
        self._mesh = None
        self.vertices = vertices
        self.faces = faces
        
        
    def apply_transform(self, transform):
        if not self._mesh:
            self._mesh = o3d.geometry.TriangleMesh()
            self._mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        
        vertices = np.asarray(self.vertices)
        vertices_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
        transformed_vertices = (transform @ vertices_h.T).T[:, :3]
        self._mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)

        self._face_normals = np.asarray(self._mesh.triangle_normals)

        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        self._raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self._mesh))

    def sample_points(self, num_points, seed=42, return_normals=False):
        if not self._mesh:
            self._mesh = o3d.geometry.TriangleMesh()
            self._mesh.triangles = o3d.utility.Vector3iVector(self.faces)
            self._mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        
        o3d.utility.random.seed(seed)
        pcd = self._mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points)
        if return_normals:
            normals = np.asarray(pcd.normals)
            return points, normals
        return points

    def bounding_box(self, padding=0.0, padding_ratio=0):
        aabb = self._mesh.get_axis_aligned_bounding_box()
        world_min = aabb.get_min_bound()
        world_max = aabb.get_max_bound()
        ranges = np.array(list(zip(world_min, world_max)))
        extents = ranges[:, 1] - ranges[:, 0]
        ranges[:, 0] -= padding + padding_ratio * extents
        ranges[:, 1] += padding + padding_ratio * extents
        world_min = ranges[:, 0]
        world_max = ranges[:, 1]
        return np.hstack([world_min, world_max])

    def compute_sdf(self, query: np.array, thresh=1e-3):
        points = query.astype(np.float32)

        closest = self._raycasting_scene.compute_closest_points(points)
        closest_points = closest["points"].numpy()

        displacement = torch.from_numpy(closest_points - query)
        distance = torch.norm(displacement, dim=-1)

        ray_destination = np.repeat(self.bounding_box(padding=1.0)[None, :3], points.shape[0], axis=0)
        ray_destination = ray_destination + 1e-4 * np.random.randn(*points.shape)
        ray_destination = ray_destination.astype(np.float32)
        direction = ray_destination - points
        rays = np.concatenate([points, direction], axis=-1).astype(np.float32)
        intersection_counts = self._raycasting_scene.count_intersections(rays).numpy()
        is_inside = intersection_counts % 2 == 1
        is_inside = torch.from_numpy(is_inside).to(distance.device)

        sdf_vals = torch.where(is_inside, -distance, distance)

        on_surface = distance < thresh
        sdf_vals = torch.where(on_surface, torch.zeros_like(sdf_vals), sdf_vals)

        return sdf_vals
    
    def compute_inside(self, query: np.array):
        if not self._mesh:
            self._mesh = o3d.geometry.TriangleMesh()
            self._mesh.triangles = o3d.utility.Vector3iVector(self.faces)
            self._mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
            
        self._raycasting_scene = o3d.t.geometry.RaycastingScene()
        self._raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(self._mesh))
        
        result = self._raycasting_scene.compute_occupancy(query).numpy() 
        
        return result
        