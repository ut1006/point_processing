import open3d as o3d
import sys

def visualize_point_cloud_with_octree(ply_file):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # OCTREEの作成
    octree = o3d.geometry.Octree(max_depth=6)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    
    # OCTREEの可視化
    o3d.visualization.draw_geometries([octree, pcd])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_point_cloud_with_octree(ply_file)
