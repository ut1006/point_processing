import open3d as o3d
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud_with_normals(ply_file):
    # Load PLY
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Calculate normal vector using KD tree.
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Get point cloud data
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    
    # Visualize point cloud with normals using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.quiver(points[:, 0], points[:, 1], points[:, 2], 
              normals[:, 0], normals[:, 1], normals[:, 2], 
              length=0.1, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Animate the view using Open3D
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False
    
    o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_point_cloud_with_normals(ply_file)
