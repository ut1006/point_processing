import open3d as o3d
import numpy as np
import sys
from sklearn.decomposition import PCA

def visualize_pca_directions(ply_file):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Octree作成
    octree = o3d.geometry.Octree(max_depth=2)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    lines = []
    points = []

    def pca_traverse(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            # Leafノードでのみ処理する
            node_points = np.asarray([pcd.points[i] for i in node.indices])
            
            # データの個数が1個以下の場合はスキップ
            if node_points.shape[0] <= 2:
                return
            
            # 点群の主成分分析
            pca = PCA(n_components=3)  
            pca.fit(node_points)

            l = pca.explained_variance_[0]
            vector = pca.components_[2]

            print(f"Explained variance: {l}")
            print(f"Principal component vector: {vector}")
            
            # 中心点を計算
            centroid = np.mean(node_points, axis=0)
            
            # 線分の始点と終点を計算
            start_point = centroid
            end_point = centroid + vector * np.sqrt(l)

            points.append(start_point)
            points.append(end_point)
            lines.append([len(points) - 2, len(points) - 1])

    # OctreeをトラバースしてPCAを計算
    octree.traverse(pca_traverse)
    print(f"Points: {points}")
    print(f"Lines: {lines}")
    
    # 線分を作成して可視化
    line_set = o3d.geometry.LineSet()
    print(len(points))
    print(len(lines))
    if points:
        line_set.points = o3d.utility.Vector3dVector(points)
        print("%")
        if lines:
            line_set.lines = o3d.utility.Vector2iVector(lines)
            print("&")
            line_set.paint_uniform_color([1, 0, 0])
            print("=")
            # 点群と線分を可視化
            o3d.visualization.draw_geometries([pcd, line_set])
        else:
            print("No lines to display.")
    else:
        print("No points to display.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_pca_directions(ply_file)
