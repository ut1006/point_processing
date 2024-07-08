import open3d as o3d
import numpy as np
import sys

def visualize_filtered_and_projected_point_cloud(ply_file, min_y, max_y):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 点群のnumpy配列への変換
    points = np.asarray(pcd.points)
    print("Loaded points:", points.shape)
    
    # Y座標がmin_y〜max_yの範囲にある点をフィルタリング
    filtered_points = points[(points[:, 1] >= min_y) & (points[:, 1] <= max_y)]
    print("Filtered points:", filtered_points.shape)
    
    # XZ平面への射影（Y成分を無視）
    projected_points = filtered_points[:, [0, 2]]
    print("Projected points:", projected_points.shape)
    
    # 射影した点群の作成
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(np.hstack((projected_points, np.zeros((projected_points.shape[0], 1)))))
    print("Projected point cloud created")
    
    # Visualizerを使って表示
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Projected Point Cloud Viewer', width=600, height=400)  # ウィンドウサイズを指定
        vis.add_geometry(projected_pcd)
        vis.run()
        vis.destroy_window()
    except Exception as e:
        print("Error during visualization:", e)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("instruction: python this_code.py <ply_file> <min_y> <max_y>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    min_y = float(sys.argv[2])
    max_y = float(sys.argv[3])
    
    visualize_filtered_and_projected_point_cloud(ply_file, min_y, max_y)
