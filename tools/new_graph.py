import numpy as np
import networkx as nx

def new_graph(granular_ball_list, graph):
    length = len(granular_ball_list)
    graph_new = nx.Graph()
    nodes = np.array(graph.nodes())

    for i in range(length):
        granular_ball = granular_ball_list[i]
        graph_new.add_node(i, label=granular_ball[-1])

    node_to_granular_ball_index = {}
    for i, granular_ball in enumerate(granular_ball_list):
        for node in granular_ball[0]:
            node_index = int(node[-1])
            node_to_granular_ball_index[nodes[node_index]] = i

    for u, v in graph.edges():
        if u in node_to_granular_ball_index and v in node_to_granular_ball_index:
            u_index = node_to_granular_ball_index[u]
            v_index = node_to_granular_ball_index[v]
            if u_index != v_index:
                graph_new.add_edge(u_index, v_index)

    return graph_new
