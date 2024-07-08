import open3d as o3d
import numpy as np
import sys

def compute_centroids_of_obbs(pcd, max_depth=6):
    # 8分木を作成
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    # 各ノードのOBBの重心を格納するリスト
    centroids = []

    def traverse(node, node_info):
        if isinstance(node, o3d.geometry.OctreeInternalNode):
            for child in node.children:
                if child is not None:
                    traverse(child, node_info)
        elif isinstance(node, o3d.geometry.OctreeLeafNode):
            # リーフノードの点群を取得
            points = np.asarray([pcd.points[i] for i in node.indices])
            if points.shape[0] < 4:
                return  # 点が4つ未満のノードはスキップ

            # OBBを計算
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
            # OBBの重心を計算
            centroid = obb.get_center()
            centroids.append(centroid)

    traverse(octree.root_node, None)
    print(len(centroids))
    return np.array(centroids)

def calculate_variance(points, centroid):
    return np.sum(np.linalg.norm(points - centroid, axis=1)**2)

def gradient_descent(pcd, learning_rate=1, iterations=3):
    # 法線を推定
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    centroid = np.mean(points, axis=0)
    
    for _ in range(iterations):
        for i in range(len(points)):
            # 現在の分散を計算
            initial_variance = calculate_variance(points, centroid)
            
            # 点を法線方向に少し移動（法線は初期推定時のまま）
            delta = learning_rate * normals[i]
            new_points = points.copy()
            new_points[i] += delta
            
            # 新しい重心を計算
            new_centroid = np.mean(new_points, axis=0)
            
            # 新しい分散を計算
            new_variance = calculate_variance(new_points, new_centroid)
            
            # 分散が減少するなら点を更新
            if new_variance < initial_variance:
                points[i] += delta
                centroid = new_centroid
    
    # 更新された点群を保存
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def save_point_cloud(pcd, file_name):
    o3d.io.write_point_cloud(file_name, pcd)

def visualize_centroids(ply_file):
    # 点群と重心点群を可視化
    o3d.visualization.draw_geometries([pcd])
    


def save_centroids_to_ply(pcd, original_file):
    # 新しいファイル名を作成
    new_file = original_file.replace(".ply", "_centered.ply")

    # PLY形式で保存
    o3d.io.write_point_cloud(new_file, pcd, write_ascii=True)

    # ヘッダーを追加
    with open(new_file, 'r') as f:
        lines = f.readlines()
    
    header = [
        "ply\n",
        "format ascii 1.0\n",
        "comment VCGLIB generated\n",
        f"element vertex {len(pcd.points)}\n",
        "property float x\n",
        "property float y\n",
        "property float z\n",
        "property uchar red\n",
        "property uchar green\n",
        "property uchar blue\n",
        "end_header\n"
    ]

    # ヘッダーを付けて保存
    with open(new_file, 'w') as f:
        f.writelines(header + lines[lines.index("end_header\n")+1:])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    pcd = o3d.io.read_point_cloud(ply_file)
    optimized_pcd = gradient_descent(pcd)
    save_point_cloud(optimized_pcd, ply_file.replace(".ply", "_optimized.ply"))
    visualize_centroids(ply_file.replace(".ply", "_optimized.ply"))
