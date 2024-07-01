import open3d as o3d
import numpy as np
import sys
from sklearn.decomposition import PCA

def visualize_pca_directions(ply_file):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Octree作成
    octree = o3d.geometry.Octree(max_depth=6)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    def pca_traverse(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode) and isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            # Leafノードでのみ処理する
            points = np.asarray([pcd.points[i] for i in node.indices])
            
            # データの個数が1個以下の場合はスキップ
            if points.shape[0] <= 1:
                return
            
            # 点群の主成分分析
            pca = PCA(n_components=1)  
            pca.fit(points)

            l = pca.explained_variance_[0]
            vector = pca.components_[0]

            print(f"Explained variance: {l}")
            print(f"Principal component vector: {vector}")
            
    # OctreeをトラバースしてPCAを計算
    octree.traverse(pca_traverse)
    
    # 点群可視化
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_pca_directions(ply_file)
