import open3d as o3d
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cartesian_to_cylindrical(points):
    cylindrical_coords = []
    for point in points:
        x, y, z = point
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        cylindrical_coords.append([r, theta, z])
    return np.array(cylindrical_coords)

def visualize_point_cloud_with_red_points(ply_file):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 点群を取得
    points = np.asarray(pcd.points)
    
    # デカルト座標を円柱座標に変換
    cylindrical_coords = cartesian_to_cylindrical(points)
    
    # rが0.5以下の点を探す
    mask = cylindrical_coords[:, 0] <= 0.5
    red_points = points[mask]
    other_points = points[~mask]
    
    # プロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # rが0.5以下の点を赤くプロット
    ax.scatter(red_points[:, 0], red_points[:, 1], red_points[:, 2], color='red', s=1, label='r <= 0.5')
    
    # それ以外の点を青くプロット
    ax.scatter(other_points[:, 0], other_points[:, 1], other_points[:, 2], color='blue', s=1, label='r > 0.5')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud with r <= 0.5 Highlighted in Red')
    ax.legend()
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_point_cloud_with_red_points(ply_file)
