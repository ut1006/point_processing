import open3d as o3d
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hdbscan

def compute_principal_components(pcd, k_neighbors):
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

def visualize_point_cloud_with_clusters(pcd, k_neighbors):
    # 主成分の計算
    principal_components = compute_principal_components(pcd, k_neighbors)
    
    # 各主成分ベクトルを正規化（単位ベクトル）
    normalized_principal_components = np.array([v / np.linalg.norm(v) for v in principal_components])
    
    # HDBSCANでクラスタリング
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = clusterer.fit_predict(normalized_principal_components)
    
    # クラスタリング結果をもとのPLY点群に色分け
    colors = plt.cm.tab20(labels / (max(labels) if max(labels) > 0 else 1))
    colors[labels == -1] = [0, 0, 0, 1]  # ノイズポイントを黒色に設定
    
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # RGBAからRGBへ
    
    # 点群を可視化
    o3d.visualization.draw_geometries([pcd], f"Clustering with k_neighbors={k_neighbors}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    pcd = o3d.io.read_point_cloud(ply_file)
    
    k_neighbors_values = [1500, 1000, 500, 100]
    
    for k in k_neighbors_values:
        visualize_point_cloud_with_clusters(pcd, k)
