import open3d as o3d
import numpy as np
import cupy as cp
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
def load_point_cloud(ply_file):
    pcd = o3d.io.read_point_cloud(ply_file)
    return pcd

def estimate_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.normals)

def find_medial_axis(points, normals):
    medial_axis_points = []
    
    # NumPy配列をCuPy配列に変換
    points_cp = cp.asarray(points)
    normals_cp = cp.asarray(normals)
    
    for i in range(len(points)):
        point = points_cp[i]
        normal = normals_cp[i]
        opposite_direction = -normal
        low, high = 0.0, 10.0  # 初期範囲を設定。適宜調整する。
        
        while high - low > 1e-6:  # 精度の設定
            radius = (low + high) / 2.0
            center = point + opposite_direction * radius
            distances = cp.linalg.norm(points_cp - center, axis=1)
            inside_points = points_cp[distances < radius]
            
            if len(inside_points) == 2:  # 自身を含む2点が球内にある
                medial_axis_points.append(cp.asnumpy(center))
                # print("!")
                break
            elif len(inside_points) > 2:
                high = radius
            else:
                low = radius
            # print(radius)
        if len(inside_points) != 2:  # 適切な半径が見つからなかった場合は初期範囲を広げる
            high += 1.0
        # time.sleep(1)
        print(f"Processed {i+1}/{len(points)} points ({(i+1)/len(points)*100:.2f}%)")

    return np.array(medial_axis_points)

def recursive_medial_axis(points, normals, iterations=3):
    for _ in range(iterations):
        medial_axis_points = find_medial_axis(points, normals)
        if len(medial_axis_points) == 0:
            break
        points = medial_axis_points
        normals = np.zeros_like(medial_axis_points)  # 再帰計算のために法線をリセット
    return points

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
    
    medial_axis_points = recursive_medial_axis(points, normals, iterations=3)
    visualize_medial_axis(points, medial_axis_points)
