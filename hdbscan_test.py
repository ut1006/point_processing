import open3d as o3d
import sys
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
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

def create_features(pcd, principal_components, vector_weight=1.0):
    # 点群の座標を取得
    points = np.asarray(pcd.points)
    
    # 座標を標準化
    scaler = StandardScaler()
    points_normalized = scaler.fit_transform(points)
    
    # ベクトル部分に重みをかける
    principal_components_weighted = principal_components * vector_weight
    
    # 6次元特徴量を作成
    features = np.hstack((points_normalized, principal_components_weighted))
    
    return features

def perform_hdbscan_clustering(features, min_samples=5):
    clustering = hdbscan.HDBSCAN(min_samples=min_samples).fit(features)
    cluster_labels = clustering.labels_
    return cluster_labels

def visualize_clusters(pcd, cluster_labels):
    # クラスタリング結果を色で表示
    unique_labels = np.unique(cluster_labels)
    colors = plt.get_cmap("tab10")(cluster_labels / max(unique_labels))
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 主成分の計算
    principal_components = compute_principal_components(pcd)
    
    # 6次元特徴量の作成（ベクトル部分に重みをかける）
    vector_weight = 2.0  # ベクトル部分の重みを2倍に設定
    features = create_features(pcd, principal_components, vector_weight)
    
    # HDBSCANクラスタリングの実行
    cluster_labels = perform_hdbscan_clustering(features, min_samples=5)
    
    # クラスタリング結果の可視化
    visualize_clusters(pcd, cluster_labels)
    
    # クラスタ数の表示
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters: {n_clusters}")
