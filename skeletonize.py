import open3d as o3d
import numpy as np
import cupy as cp
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_point_cloud(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    return pcd

def estimate_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.normals)

def find_medial_axis(points, normals):
    medial_axis_points = []
    radius = 1.0
    
    # NumPy配列をCuPy配列に変換
    points_cp = cp.asarray(points)
    normals_cp = cp.asarray(normals)
    
    for i in range(len(points)):
        point = points_cp[i]
        normal = normals_cp[i]
        opposite_direction = normal
        print(i/len(points), "%")
        while True:
            center = point + opposite_direction * radius
            distances = cp.linalg.norm(points_cp - center, axis=1)
            inside_points = points_cp[distances < radius]
            
            if len(inside_points) == 2: # including myself, 2 points are in the sphere.
                medial_axis_points.append(cp.asnumpy(center))
                break
            if len(inside_points) > 2:
                radius -= 0.0001
            if len(inside_points) < 2:
                radius += 0.000001
    
    return np.array(medial_axis_points)

def visualize_medial_axis(points, medial_axis_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='k')  # 元の点群を黒色でプロット
    ax.scatter(medial_axis_points[:, 0], medial_axis_points[:, 1], medial_axis_points[:, 2], s=1, color='r')  # 中心軸の点群を赤色でプロット
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    pcd = load_point_cloud(ply_file)
    points = np.asarray(pcd.points)
    normals = estimate_normals(pcd)
    medial_axis_points = find_medial_axis(points, normals)
    visualize_medial_axis(points, medial_axis_points)
