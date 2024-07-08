import open3d as o3d
import numpy as np
import sys
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.DEBUG)

def load_point_cloud(ply_file):
    logging.debug(f"Loading point cloud from {ply_file}")
    pcd = o3d.io.read_point_cloud(ply_file)
    return pcd

def estimate_normals(pcd):
    logging.debug("Estimating normals")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.normals)

def build_octree(points, max_depth=10):
    logging.debug("Building octree")
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)), size_expand=0.01)
    return octree

def process_point(i, points, normals, octree):
    logging.debug(f"Processing point index {i}")
    medial_axis_points = []
    point = points[i]
    normal = normals[i]
    radius = 1.0
    opposite_direction = normal

    while True:
        center = point + opposite_direction * radius
        leaf_node = octree.locate_leaf_node(center)
        if leaf_node is None:
            radius += 0.000001
            continue
        indices = leaf_node.indices
        inside_points = points[indices]

        if len(inside_points) == 2:  # including myself, 2 points are in the sphere.
            medial_axis_points.append(center)
            break
        if len(inside_points) > 2:
            radius -= 0.0001
        if len(inside_points) < 2:
            radius += 0.000001

    return medial_axis_points

def find_medial_axis(points, normals, octree):
    logging.debug("Finding medial axis")
    medial_axis_points = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_point, i, points, normals, octree) for i in range(len(points))]
        for future in futures:
            result = future.result()
            if result:  # Check if the result is not empty
                medial_axis_points.extend(result)

    return np.array(medial_axis_points)

def visualize_medial_axis(points, medial_axis_points):
    logging.debug("Visualizing medial axis")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='k')  # Plot original point cloud in black
    ax.scatter(medial_axis_points[:, 0], medial_axis_points[:, 1], medial_axis_points[:, 2], s=1, color='r')  # Plot medial axis points in red
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    logging.debug(f"Starting process for {ply_file}")
    pcd = load_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    normals = estimate_normals(pcd)
    octree = build_octree(points)
    medial_axis_points = find_medial_axis(points, normals, octree)
    visualize_medial_axis(points, medial_axis_points)
