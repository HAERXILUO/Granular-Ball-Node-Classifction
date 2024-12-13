import numpy as np
from math import sqrt
ini = float('inf')
import networkx as nx

def initial_splite(C, graph, id_dict, id_dict_oldtonew, labels, total_degree_dict):

    connected_components = list(nx.connected_components(graph))
    new_clusters = []
    for i, component in enumerate(connected_components, start=1):

        subgraph = graph.subgraph(list(component))
        new_node_ids = []
        for old_id in list(component):
            new_node_ids.append(id_dict_oldtonew[old_id])

        component_data = [C[0][0][i] for i in new_node_ids]
        component_degree_dict = {node: total_degree_dict[node] for node in new_node_ids}
        component_C = [component_data, component_degree_dict]

        #select center point
        centers = select_initial_centers(component_C, component_degree_dict)

        if len(centers) != 0:
            Distances = []
            for center in centers:
                Distance = nx.single_source_shortest_path_length(subgraph, id_dict[center])
                Distances.append(Distance)

            balls_data_nodes = [[] for _ in centers]
            balls_degree_dict = [{} for _ in centers]
            for old_id in list(component):
                min = np.inf
                min_center_idx = -1

                for Distance in Distances:
                    if Distance[old_id] < min:
                        min = Distance[old_id]
                        min_center_idx = next(iter(Distance.keys()))
                center_index = centers.index(id_dict_oldtonew[min_center_idx])
                balls_data_nodes[center_index].append(C[0][0][id_dict_oldtonew[old_id]])

            for index, center in enumerate(centers):
                ball_nodes = balls_data_nodes[index]
                ball_nodes_index = [int(row[-1]) for row in ball_nodes]

                for node in ball_nodes_index:
                    balls_degree_dict[index][node] = total_degree_dict[node]

            balls = [
                [balls_data_nodes[i], balls_degree_dict[i]]
                for i in range(len(centers))
            ]
            for ball in balls:
                new_clusters.append(ball)
    return new_clusters


def select_initial_centers(component_C, degree_dict):
    node_info = component_C[0]
    class_nodes_dict = {}

    for info in node_info:
        label = info[-2]
        node_index = int(info[-1])
        if label not in class_nodes_dict and label != -1:
            class_nodes_dict[label] = []
        if label != -1:
            class_nodes_dict[label].append(node_index)

    num_classes = len(class_nodes_dict)
    num_nodes = len(node_info)

    if num_classes != 0:
        centers_per_class = max(1, int(sqrt(num_nodes) / num_classes))

    centers = []
    for label, nodes in class_nodes_dict.items():
        degrees = [(node, degree_dict[node]) for node in nodes if node in degree_dict]
        degrees.sort(key=lambda x: x[1], reverse=True)
        selected_centers = [node for node, _ in degrees[:centers_per_class]]
        centers.extend(selected_centers)
    return centers


