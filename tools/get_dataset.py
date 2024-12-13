import numpy as np
import networkx as nx


def get_dataset(data_points, adjacency_matrix,  data_labels, seed=None,):
    graph = nx.Graph()

    for node_id, point in enumerate(data_points):
        graph.add_node(node_id, attributes=point)

    for node_id, label in enumerate(data_labels):
        graph.nodes[node_id]['label'] = label

    adjacency_matrix = adjacency_matrix.tocsr()

    rows, cols = adjacency_matrix.nonzero()

    for i, j in zip(rows, cols):
        if i != j:
            graph.add_edge(i, j, edge_attr=np.zeros(3))

    return graph



