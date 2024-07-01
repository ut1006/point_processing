import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA

def get_points_within_voxel(pcd, voxel_center, voxel_size):
    points = np.asarray(pcd.points)
    min_bound = voxel_center - voxel_size / 2
    max_bound = voxel_center + voxel_size / 2
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    return points[mask]

def estimate_tree_skeleton(pcd, max_depth=4):
    # Octreeで点群を分割
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    # 各ボクセルの中心点と主成分方向を格納するリスト
    centers = []
    directions = []

    # Octreeを再帰的に巡り、各ノードでPCAを実行
    def traverse_and_analyze(node, depth, voxel_center, voxel_size):
        if isinstance(node, o3d.geometry.OctreeLeafNode):  # 葉ノードかどうかをチェック
            points_in_voxel = get_points_within_voxel(pcd, voxel_center, voxel_size)
            if points_in_voxel.shape[0] >= 3:  # PCAには最低3点必要
                pca = PCA(n_components=3)
                pca.fit(points_in_voxel)
                center = np.mean(points_in_voxel, axis=0)
                direction = pca.components_[0]  # 第1主成分
                centers.append(center)
                directions.append(direction)
        elif isinstance(node, o3d.geometry.OctreeInternalNode):
            child_voxel_size = voxel_size / 2
            for i, child in enumerate(node.children):
                if child is not None:
                    child_voxel_center = voxel_center + child_voxel_size * np.array([
                        (i & 1) * 2 - 1,
                        ((i >> 1) & 1) * 2 - 1,
                        ((i >> 2) & 1) * 2 - 1
                    ])
                    traverse_and_analyze(child, depth + 1, child_voxel_center, child_voxel_size)

    root_voxel_center = np.mean(np.asarray(pcd.get_min_bound()) + np.asarray(pcd.get_max_bound())) / 2
    root_voxel_size = np.max(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    traverse_and_analyze(octree.root_node, 0, root_voxel_center, root_voxel_size)

    # 骨格をラインセットとして構築
    lines = []
    for i, center in enumerate(centers):
        start = center
        end = center + directions[i] * 10  # 方向に沿って線を引く
        lines.append([start, end])

    # ラインセットをOpen3Dで作成
    points = []
    for start, end in lines:
        points.append(start)
        points.append(end)

    lines_indices = [[i, i + 1] for i in range(0, len(points), 2)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines_indices))

    # Open3Dで可視化
    print(len(line_set.points))
    o3d.visualization.draw_geometries([pcd, line_set])

# PLYファイルの読み込み
ply_file = "tree.ply"  # 正しいファイル名と拡張子を指定してください
pcd = o3d.io.read_point_cloud(ply_file)

if not pcd.is_empty():
    # 樹木の骨格を推定して可視化
    estimate_tree_skeleton(pcd)
else:
    print("Point cloud is empty or failed to load.")
