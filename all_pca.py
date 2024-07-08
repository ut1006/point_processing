import open3d as o3d
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_principal_components(pcd, k_neighbors=100):
    # KDTreeを使用して点群の近傍点を検索
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 各点の主成分を格納するリスト
    principal_components = []

    for i in range(len(pcd.points)):
        # 現在の点を取得
        point = np.asarray(pcd.points)[i]
        
        # 近傍点を検索
        [_, idx, _] = kdtree.search_knn_vector_3d(point, k_neighbors)
        
        # 近傍点の座標を取得
        neighbors = np.asarray(pcd.points)[idx, :]
        
        # 主成分分析を実行
        cov_matrix = np.cov(neighbors.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 主成分を格納
        principal_components.append(eigenvectors[:, -1])  # 最大の固有値に対応する固有ベクトル

    return np.array(principal_components)

def visualize_point_cloud_with_principal_components(ply_file):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 主成分の計算
    principal_components = compute_principal_components(pcd)
    
    # 主成分を可視化するためのラインを追加
    lines = []
    for i, point in enumerate(pcd.points):
        start_point = point
        end_point = point + principal_components[i] * 0.1  # 主成分ベクトルをスケール
        lines.append([start_point, end_point])
    
    # ラインセットを作成
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.vstack([pcd.points, np.array(lines)[:, 1]]))
    line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i + len(pcd.points)] for i in range(len(pcd.points))]))
    
    # 点群とラインセットを可視化
    o3d.visualization.draw_geometries([pcd, line_set])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_point_cloud_with_principal_components(ply_file)
