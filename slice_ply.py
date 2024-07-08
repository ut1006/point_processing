import open3d as o3d
import numpy as np
import sys
import tkinter as tk

def visualize_filtered_points_with_color(ply_file, min_y, max_y):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 点群のnumpy配列への変換
    points = np.asarray(pcd.points)
    
    # 元の点群の色を取得（存在しない場合は白に設定）
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(points)  # 初期色はすべての点を白に設定
    
    # Y座標がmin_y〜max_yの範囲にある点のフィルタリング
    mask = (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    colors[mask] = [1, 0, 0]  # フィルタリングした点を赤色に設定
    
    # 色を点群に適用
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Tkinterウィンドウを作成して位置を調整
    root = tk.Tk()
    root.withdraw()  # メインウィンドウを隠す
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 600
    window_height = 400
    x_offset = 600
    window_x = x_offset
    window_y = 0  # 垂直方向に中央に配置
    
    # Visualizerオブジェクトの作成
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud Viewer', width=window_width, height=window_height, left=window_x, top=window_y)
    
    # 点群を追加して可視化
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    root.destroy()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("instruction: python this_code.py <ply_file> <min_y> <max_y>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    min_y = float(sys.argv[2])
    max_y = float(sys.argv[3])
    
    visualize_filtered_points_with_color(ply_file, min_y, max_y)
