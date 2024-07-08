import open3d as o3d
import numpy as np
import sys

# 点群の法線ベクトルを計算
def compute_normals(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=300))
    return np.asarray(pcd.normals)

# Octreeによる点群の分割
def build_octree(pcd, max_depth=4):
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    return octree

# 分割領域内の点を法線ベクトルに沿って移動させて分散を最小化
def minimize_variance(points, normals, learning_rate=0.01, iterations=500):
    for _ in range(iterations):
        for i in range(len(points)):
            point = points[i]
            normal = normals[i]
            gradient = 2 * (point - np.mean(points, axis=0))
            points[i] = point - learning_rate * np.dot(gradient, normal) * normal
    return points

# Octreeの各分割領域内の点を処理
def process_octree_nodes(node, points, normals):
    if isinstance(node, o3d.geometry.OctreeInternalNode):
        for child in node.children:
            if child is not None:
                process_octree_nodes(child, points, normals)
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        indices = node.indices
        if len(indices) > 0:
            sub_points = points[indices]
            sub_normals = normals[indices]
            new_points = minimize_variance(sub_points, sub_normals)
            points[indices] = new_points
    return points

# 点群を読み込んで処理するメイン関数
def main(ply_file):
    # 点群の読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 法線ベクトルの計算
    normals = compute_normals(pcd)
    
    # Octreeの構築
    octree = build_octree(pcd)
    
    # 点群の座標と色を取得
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Octreeのルートノードを取得し、各ノードを処理
    root_node = octree.root_node
    new_points = process_octree_nodes(root_node, points, normals)
    
    # 新しい点群を保存
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_points)
    new_pcd.colors = o3d.utility.Vector3dVector(colors)
    new_pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # 結果を表示
    o3d.visualization.draw_geometries([new_pcd])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    main(ply_file)
