import open3d as o3d
import sys
import numpy as np

def compute_principal_components(pcd, indices=None, k_neighbors=100):
    # KDTreeを使用して点群の近傍点を検索
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    if indices is None:
        indices = np.arange(len(pcd.points))

    # 各点の主成分を格納するリスト
    principal_components = np.zeros((len(pcd.points), 3))

    for i in indices:
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
        principal_components[i] = eigenvectors[:, -1]  # 最大の固有値に対応する固有ベクトル

    return principal_components

def is_principal_component_aligned(principal_components, threshold=0.7):
    # 各ベクトルを正規化
    principal_components /= np.linalg.norm(principal_components, axis=1)[:, np.newaxis]
    
    # ベクトルの向きを揃える（z成分が負のベクトルを反転）
    principal_components[principal_components[:, 2] < 0] *= -1
    
    # 平均ベクトルを計算
    mean_vector = np.mean(principal_components, axis=0)
    mean_vector_norm = np.linalg.norm(mean_vector)
    
    # 平均ベクトルのノルムがthreshold以上かを確認
    return mean_vector_norm >= threshold, mean_vector_norm

def visualize_point_cloud_with_octree(ply_file, depth=6, min_points=6, threshold=0.9, k_neighbors=500):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 主成分の計算
    principal_components = compute_principal_components(pcd)
    
    # OCTREEの作成
    octree = o3d.geometry.Octree(max_depth=depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    
    # 一致している点を特定するためのインデックスリスト
    aligned_indices = []

    def process_node(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode) and len(node.indices) >= min_points:
            # 分割内の点を取得
            points_indices = node.indices
            node_principal_components = principal_components[points_indices]
            
            # 主成分の一致度を確認
            aligned, norm = is_principal_component_aligned(node_principal_components, threshold)
            if aligned:
                aligned_indices.extend(points_indices)

    # オクツリーをトラバースして各分割の主成分ベクトルを確認
    octree.traverse(process_node)
    
    # 一致している点を赤色で表示
    colors = np.zeros((len(pcd.points), 3))
    colors[:] = [0, 1, 0]  # 初期設定は緑
    colors[aligned_indices] = [1, 0, 0]  # 一致している点を赤色に変更

    # 一致していない点群を新しい点群として取得
    remaining_points_indices = np.setdiff1d(np.arange(len(pcd.points)), aligned_indices)
    
    # 残りの点群に対して主成分分析を再実施
    remaining_principal_components = compute_principal_components(pcd, indices=remaining_points_indices, k_neighbors=k_neighbors)

    # 残りの点群に対してオクツリーの再作成
    remaining_octree = o3d.geometry.Octree(max_depth=depth)
    remaining_octree.convert_from_point_cloud(pcd.select_by_index(remaining_points_indices), size_expand=0.01)

    # 青く塗る点のインデックスリスト
    blue_indices = []

    def process_remaining_node(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode) and len(node.indices) >= min_points:
            # 分割内の点を取得
            points_indices = node.indices
            node_principal_components = remaining_principal_components[points_indices]
            
            # 主成分の一致度を確認
            aligned, norm = is_principal_component_aligned(node_principal_components, threshold)
            if aligned:
                blue_indices.extend(points_indices)

    # 残りの点群のオクツリーをトラバースして各分割の主成分ベクトルを確認
    remaining_octree.traverse(process_remaining_node)

    # 一致している点を青色で表示
    blue_indices_global = remaining_points_indices[blue_indices]
    colors[blue_indices_global] = [0, 0, 1]  # 一致している点を青色に変更

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 主成分ベクトルを可視化するためのラインセットを作成
    def create_lineset_for_indices(indices, principal_components, color):
        lines = []
        points = np.asarray(pcd.points)
        for i in indices:
            start_point = points[i]
            end_point = start_point + principal_components[i] * 0.1  # スケール調整
            lines.append([start_point, end_point])
        
        line_points = np.vstack([line for line in lines])
        line_indices = [[i, i + 1] for i in range(0, len(line_points), 2)]
        
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector(line_indices)
        )
        line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(line_indices), 1)))
        return line_set

    # 赤色のラインセットを作成
    red_lineset = create_lineset_for_indices(aligned_indices, principal_components, [1, 0, 0])

    # 青色のラインセットを作成
    blue_lineset = create_lineset_for_indices(blue_indices_global, remaining_principal_components, [0, 0, 1])

    # 点群とラインセットを可視化
    o3d.visualization.draw_geometries([pcd, red_lineset, blue_lineset])

    # 一致度が高かった点の数を表示
    print(f"Number of aligned points: {len(aligned_indices)}")
    print(f"Number of secondary aligned points: {len(blue_indices_global)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_point_cloud_with_octree(ply_file)
