import open3d as o3d
import numpy as np

line_set = o3d.geometry.LineSet()

points = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [3, 2, 0],
    [1, 4, 0],
    [-1,2, 0],
])
line_set.points = o3d.utility.Vector3dVector(points)

lines = np.array([
    [0, 2],
    [2, 4],
    [4, 1],
    [1, 3],
    [3, 0],
])
line_set.lines = o3d.utility.Vector2iVector(lines)

line_set.paint_uniform_color([1, 0, 0])

# o3d.visualization.draw_geometries([line_set])




line_set2 = o3d.geometry.LineSet()

points2 = np.array([
    [1, 0, 0],
    [2, 0, 1],
    [3, 2, 1],
    [2, 4, 0],
    [0,2, 0],
])
line_set2.points = o3d.utility.Vector3dVector(points2)

lines2 = np.array([
    [0, 2],
    [2, 4],
    [4, 1],
    [1, 3],
    [3, 0],
])
line_set2.lines = o3d.utility.Vector2iVector(lines2)

line_set2.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([line_set2]+[line_set])