import open3d as o3d
import numpy as np
import sys
import os
import pandas as pd
import random

class Tree_node:
    def __init__(self, pos_x, pos_y, pos_z):
        self.x = pos_x
        self.y = pos_y
        self.z = pos_z
        self.pos = np.array([self.x, self.y, self.z])

    def __repr__(self):
        return self.pos.__repr__()

class Attraction_point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.pos = np.array([self.x, self.y, self.z])

    def __repr__(self):
        return self.pos.__repr__()

class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = [self.root]
        self.transition_map = {}

    def add_child(self, parent, child):
        if child in self.nodes:
            raise ValueError("Child already exists in the tree")
        if parent in self.nodes:
            self.transition_map[child] = parent
            self.nodes.append(child)
        else:
            raise ValueError("Parent does not exist in the tree")

    def is_leaf(self, node):
        if node not in self.nodes:
            raise ValueError("Node does not exist in the tree")
        return node not in self.transition_map.values()

    def num_children(self, node):
        if node not in self.nodes:
            raise ValueError("Node does not exist in the tree")
        return list(self.transition_map.values()).count(node)

    def get_children(self, parent):
        if parent not in self.nodes:
            raise ValueError("Parent does not exist in the tree")
        return [child for child in self.transition_map if self.transition_map[child] == parent]

    def get_level(self, node):
        if node not in self.nodes:
            raise ValueError("Node does not exist in the tree")
        if node == self.root:
            return 0
        x = self.transition_map[node]
        level = 1
        while x != self.root:
            x = self.transition_map[x]
            if self.num_children(x) > 1:
                level += 1
        return level

class Simulation:
    def __init__(self, crown_attraction_points, radius_of_influence, kill_distance, D, branch_min_width):
        self.d_i = radius_of_influence
        self.d_k = kill_distance
        self.D = D
        self.iter_num = 0

        # attraction points
        x, y, z = crown_attraction_points
        attraction_pts = [Attraction_point(i, j, k) for i, j, k in zip(x, y, z)]

        # nodes
        self.nodes = []
        root = Tree_node(0, 0, 0)
        self.nodes.append(root)

        # closest node to each attraction pt
        self.closest_node = {attr_pt: None for attr_pt in attraction_pts}
        self.closest_dist = {attr_pt: np.inf for attr_pt in attraction_pts}
        self._update_closest_node(self.nodes[0])

        # branches
        self.branches = []
        self.branching_tree = Tree(root)
        self.branch_min_width = branch_min_width
        self.branch_width = {}

    def _update_closest_node(self, node):
        kill_candidates = []
        for attr_pt in self.closest_node:
            old_smallest = self.closest_dist[attr_pt]
            dist = np.linalg.norm(attr_pt.pos - node.pos)
            if dist < self.d_k:
                kill_candidates.append(attr_pt)
                continue
            if dist < self.d_i and dist < old_smallest:
                self.closest_node[attr_pt] = node
                self.closest_dist[attr_pt] = dist
        for attr_pt in kill_candidates:
            del self.closest_node[attr_pt]
            del self.closest_dist[attr_pt]

    def branch_thickness(self, node):
        if node in self.branch_width:
            return self.branch_width[node]
        if self.branching_tree.is_leaf(node):
            self.branch_width[node] = self.branch_min_width
            return self.branch_min_width
        if self.branching_tree.num_children(node) == 1:
            w = self.branch_thickness(self.branching_tree.get_children(node)[0])
            self.branch_width[node] = w
            return w
        w = 0
        for child in self.branching_tree.get_children(node):
            w += np.square(self.branch_thickness(child))
        w = np.sqrt(w)
        self.branch_width[node] = w
        return w

    def run(self, num_iteration):
        for i in range(num_iteration):
            self._iter()
            if len(self.closest_node) == 0:
                break
        self.render_results()

    def render_results(self):
        lines = []
        colors = []
        for branch in self.branches:
            start, end, node = branch
            lines.append([start, end])
            colors.append([0.13, 0.55, 0.13])  # forestgreen color

        line_set = o3d.geometry.LineSet()
        points = np.array([node.pos for node in self.nodes])
        point_indices = {tuple(node.pos): i for i, node in enumerate(self.nodes)}
        lines = np.array([[point_indices[tuple(branch[0])], point_indices[tuple(branch[1])]] for branch in self.branches])
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set

    def _iter(self):
        self.iter_num += 1
        meta_nodes = []
        for node in self.nodes:
            S_v = {attr_pt for attr_pt in self.closest_node if self.closest_node[attr_pt] == node}
            if len(S_v) != 0:
                n = np.array([0, 0, 0], dtype=float)
                for attr_pt in S_v:
                    n += (attr_pt.pos - node.pos) / np.linalg.norm(attr_pt.pos - node.pos)
                n = n / np.linalg.norm(n)
                new_pos = node.pos + n * self.D
                new_node = Tree_node(new_pos[0], new_pos[1], new_pos[2])
                self._update_closest_node(new_node)
                branch = (node.pos, new_pos, new_node)
                self.branches.append(branch)
                self.branching_tree.add_child(node, new_node)
                meta_nodes.append(new_node)
        self.nodes.extend(meta_nodes)

    def save_transition_map_with_thickness_to_csv(self, base_filename='tree'):
        # Find the next available filename
        i = 1
        while os.path.exists(f'{base_filename}{i}.csv'):
            i += 1
        filename = f'{base_filename}{i}.csv'
        
        # Print the filename
        print(f'Saving transition map to {filename}')
        
        # Create DataFrame to store data
        data = []
        for child, parent in self.branching_tree.transition_map.items():
            thickness = self.branch_thickness(child)
            data.append([
                parent.pos[0], parent.pos[1], parent.pos[2], 
                child.pos[0], child.pos[1], child.pos[2], 
                thickness
            ])
        
        # Add leaves
        data += self.generate_leaf_data(num_leaves=300)
        
        # Convert data to DataFrame
        df = pd.DataFrame(data, columns=['Parent_x', 'Parent_y', 'Parent_z', 'Child_x', 'Child_y', 'Child_z', 'Thickness'])
        
        # Drop duplicates
        df.drop_duplicates(inplace=True)
        
        # Save DataFrame to CSV
        df.to_csv(filename, index=False, float_format='%.6f')

    def generate_leaf_data(self, num_leaves):
        leaf_data = []
        while len(leaf_data) < num_leaves:
            node = random.choice(self.nodes)
            if 0.5 <= node.pos[1] <= 3.5 and node != self.nodes[0]:  
                direction = np.random.normal(size=3)
                direction = direction / np.linalg.norm(direction) * 0.1
                leaf_end = node.pos + direction
                leaf_data.append([
                    node.pos[0], node.pos[1], node.pos[2], 
                    leaf_end[0], leaf_end[1], leaf_end[2], 
                    -1
                ])
        return leaf_data

def load_point_cloud_from_ply(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    return pcd

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_ply_file>")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    pcd = load_point_cloud_from_ply(ply_file)
    points = np.asarray(pcd.points)
    x_crown, y_crown, z_crown = points[:, 0], points[:, 1], points[:, 2]

    sim = Simulation(crown_attraction_points=(x_crown, y_crown, z_crown), radius_of_influence=0.3, kill_distance=0.2, D=0.15, branch_min_width=1)
    sim.run(50)
    
    line_set = sim.render_results()
    
    o3d.visualization.draw_geometries([pcd, line_set])
    sim.save_transition_map_with_thickness_to_csv()
