import open3d as o3d
import numpy as np
import sys
from sklearn.decomposition import PCA

def visualize_pca_directions(ply_file):
    # Load the PLY file
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Create Octree
    octree = o3d.geometry.Octree(max_depth=5)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)

    lines = []
    points = []
    centers = []
    spheres = []

    def pca_traverse(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            # Process only leaf nodes
            node_points = np.asarray([pcd.points[i] for i in node.indices])
            
            # Skip if there are 1 or fewer points
            if node_points.shape[0] <= 2:
                return
            
            # Perform PCA on the points
            pca = PCA(n_components=3)
            pca.fit(node_points)

            l = pca.explained_variance_[0]
            vector = pca.components_[0]
            print(f"explain var: {l}")
            print(f"vector: {vector}")
            # Calculate the centroid
            centroid = np.mean(node_points, axis=0)
            centers.append(centroid)
            
            # Create a small sphere at the centroid for visualization
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(centroid)
            sphere.paint_uniform_color([0, 1, 0])
            spheres.append(sphere)
            
            # Calculate the start and end points of the line
            start_point = centroid
            end_point = centroid + vector * np.sqrt(l)

            points.append(start_point)
            points.append(end_point)
            lines.append([len(points) - 2, len(points) - 1])

    # Traverse the Octree and compute PCA
    octree.traverse(pca_traverse)

    # Create lineset for the PCA directions
    line_set = o3d.geometry.LineSet()
    if points:
        line_set.points = o3d.utility.Vector3dVector(points)
        if lines:
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.paint_uniform_color([1, 0, 0])
    
    # Visualize the point cloud, PCA directions, and center points
    geometries = [pcd, line_set] + spheres
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("instruction: python this_code.py <ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    visualize_pca_directions(ply_file)
