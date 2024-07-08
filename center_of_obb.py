import open3d as o3d
import sys
import numpy as np

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

def visualize_centroids(ply_file):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)

    # OBBの重心を計算
    centroids = compute_centroids_of_obbs(pcd)

    # 重心点群を作成
    centroid_pcd = o3d.geometry.PointCloud()
    centroid_pcd.points = o3d.utility.Vector3dVector(centroids)
    centroid_pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(centroids), 3)))  # 全ての点の色を黒に設定

    # 点群と重心点群を可視化
    o3d.visualization.draw_geometries([centroid_pcd])
    
    # 重心点群を保存
    save_centroids_to_ply(centroid_pcd, ply_file)

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
    visualize_centroids(ply_file)
