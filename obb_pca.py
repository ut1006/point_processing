import open3d as o3d
import sys
import numpy as np

def compute_oriented_bounding_box(points):
    # numpy配列をopen3dのVector3dVectorに変換
    points_o3d = o3d.utility.Vector3dVector(points)
    
    # 点群からOBBを計算する
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(points_o3d)
    obb = aabb.get_oriented_bounding_box()
    obb=aabb
    return obb

def visualize_point_cloud_with_octree(ply_file):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # OCTREEの作成
    octree = o3d.geometry.Octree(max_depth=6)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    
    obbs = []

    def add_obbs(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode) and isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            points = np.asarray([pcd.points[i] for i in node.indices])
            if points.shape[0] < 2:
                return  # 点が少ない場合はスキップ
            
            obb = compute_oriented_bounding_box(points)
            obbs.append(obb)

    # オクツリーをトラバースして各OBBを計算・追加
    octree.traverse(add_obbs)
    
    # 法線の推定
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 点群とOBBを可視化
    o3d.visualization.draw_geometries([pcd] + obbs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_point_cloud_with_octree(ply_file)
