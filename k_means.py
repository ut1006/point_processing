import open3d as o3d
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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

def optimal_clustering(features):
    silhouette_scores = []
    range_n_clusters = range(2, 11)

    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, n_init=10)  # n_initを明示的に指定
        cluster_labels = clusterer.fit_predict(features)
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    
    return optimal_n_clusters, silhouette_scores

def visualize_clusters(pcd, cluster_labels):
    # クラスタリング結果を色で表示
    colors = plt.get_cmap("tab10")(cluster_labels / max(cluster_labels))
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
    
    # 最適なクラスタ数の決定
    optimal_n_clusters, silhouette_scores = optimal_clustering(features)
    print(optimal_n_clusters)
    
    # クラスタリングの実行
    clusterer = KMeans(n_clusters=optimal_n_clusters, n_init=10)  # n_initを明示的に指定
    cluster_labels = clusterer.fit_predict(features)
    
    # クラスタリング結果の可視化
    visualize_clusters(pcd, cluster_labels)
    
    # シルエットスコアのプロット
    plt.figure()
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of clusters')
    plt.show()
