import open3d as o3d
import numpy as np
import sys

def load_point_cloud(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    return np.asarray(pcd.points)

def save_point_cloud(points, output_file):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_file, pcd)

def compute_bounding_box(points):
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    return min_bound, max_bound

def divide_into_voxels(points, voxel_size):
    min_bound, max_bound = compute_bounding_box(points)
    voxel_grid = np.floor((points - min_bound) / voxel_size).astype(int)
    unique_voxels = np.unique(voxel_grid, axis=0)
    return unique_voxels, voxel_grid, min_bound

def assign_points_to_voxels(points, unique_voxels, voxel_grid):
    voxel_dict = {tuple(voxel): [] for voxel in unique_voxels}
    for point, voxel in zip(points, voxel_grid):
        voxel_dict[tuple(voxel)].append(point)
    return voxel_dict

def compute_potential(voxel_dict, min_bound, voxel_size, r):
    potential = np.zeros((len(voxel_dict),))
    unique_voxels = list(voxel_dict.keys())
    for i, voxel in enumerate(unique_voxels):
        voxel_center = min_bound + (np.array(voxel) + 0.5) * voxel_size
        points_in_voxel = np.array(voxel_dict[voxel])
        if points_in_voxel.size > 0:
            distances = np.linalg.norm(points_in_voxel - voxel_center, axis=1)
            potential[i] = np.sum(1 - (distances / r) ** 2)
        else:
            potential[i] = 0

    potential = (potential - np.min(potential)) / (np.max(potential) - np.min(potential))
    return unique_voxels, potential

def compute_gradient(unique_voxels, potential, voxel_size):
    gradients = np.zeros((len(unique_voxels), 3), dtype=float)
    for i, voxel in enumerate(unique_voxels):
        for j in range(3):
            neighbor_voxel = list(voxel)
            neighbor_voxel[j] += 1
            if tuple(neighbor_voxel) in unique_voxels:
                neighbor_idx = unique_voxels.index(tuple(neighbor_voxel))
                gradients[i, j] = potential[neighbor_idx] - potential[i]
            else:
                gradients[i, j] = -potential[i]
    return gradients

def move_points(points, voxel_dict, unique_voxels, gradients, voxel_grid, min_bound, voxel_size, p, r_effective):
    moved_points = points.copy()
    unique_voxel_dict = {tuple(voxel): i for i, voxel in enumerate(unique_voxels)}

    for i, point in enumerate(points):
        voxel = tuple(voxel_grid[i])
        voxel_idx = unique_voxel_dict[voxel]
        gradient = gradients[voxel_idx]
        direction = gradient / np.linalg.norm(gradient)
        move_distance = voxel_size * p * np.linalg.norm(gradient)
        move_vector = direction * move_distance

        neighbors = points[np.linalg.norm(points - point, axis=1) < r_effective]
        repulsion_vector = np.zeros(3)
        for neighbor in neighbors:
            if np.linalg.norm(point - neighbor) < r_effective:
                repulsion_vector += (point - neighbor) * (2 * r_effective - np.linalg.norm(point - neighbor)) / np.linalg.norm(point - neighbor)

        moved_points[i] += move_vector + repulsion_vector

    return moved_points

def main(ply_file, voxel_size=0.001, r=0.01, p=2, r_effective=0.0001, tol=1e-6):
    points = load_point_cloud(ply_file)
    unique_voxels, voxel_grid, min_bound = divide_into_voxels(points, voxel_size)
    
    print(f"Number of unique voxels: {len(unique_voxels)}")
    print(f"Unique voxels: {unique_voxels}")

    voxel_dict = assign_points_to_voxels(points, unique_voxels, voxel_grid)
    
    while True:
        unique_voxels, potential = compute_potential(voxel_dict, min_bound, voxel_size, r)
        gradients = compute_gradient(unique_voxels, potential, voxel_size)
        new_points = move_points(points, voxel_dict, unique_voxels, gradients, voxel_grid, min_bound, voxel_size, p, r_effective)

        if np.linalg.norm(new_points - points) < tol:
            break

        points = new_points

    output_file = "moved_" + ply_file
    save_point_cloud(points, output_file)
    pcd = o3d.io.read_point_cloud(output_file)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    main(ply_file)
