import open3d as o3d
import sys
import numpy as np
import os

def remove_duplicate_points_and_save(ply_file):
    # PLYファイルの読み込み
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # 点群データをnumpy配列に変換
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    # 重複点を削除
    unique_points, indices = np.unique(points, axis=0, return_index=True)
    unique_colors = np.ones_like(unique_points) * 0  # 白色を設定する

    # 新しい点群を作成
    unique_pcd = o3d.geometry.PointCloud()
    unique_pcd.points = o3d.utility.Vector3dVector(unique_points)
    if colors is not None:
        unique_pcd.colors = o3d.utility.Vector3dVector(unique_colors)

    # 新しいファイル名を作成
    base, ext = os.path.splitext(ply_file)
    output_file = f"{base}_unified{ext}"

    # 新しいPLYファイルとして保存
    o3d.io.write_point_cloud(output_file, unique_pcd, write_ascii=True)

    print(f"Saved deduplicated point cloud to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_duplicate_points_and_save.py <input_ply_file>")
        sys.exit(1)
    
    input_ply_file = sys.argv[1]
    remove_duplicate_points_and_save(input_ply_file)
