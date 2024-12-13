from tools import *
import numpy as np
import networkx as nx
import matplotlib
from tools.add_noise import add_noise
matplotlib.use('TkAgg')
import scipy.sparse as sp
from tools.split_ball_purity import split_ball_purity
from tools.purification import purification
from tools.add_id import add_id
from tools.new_graph import new_graph
from tools.corse_split import initial_splite
from tools.add_purity import add_purity
from tools.split_ball_purity import split_ball_further



def gb_division(data, args):
    seed = 0

    #whether to add noise
    if args.noise == 1:
        data = add_noise(data)
    C = []

    print(data)
    data_file = data.x.numpy()
    total_balls_num = int(args.ball_r * len(data_file))
    adjacency_matrix = sp.coo_matrix((np.ones(data.edge_index.shape[1]), (data.edge_index[0], data.edge_index[1])),
                                     shape=(data.y.shape[0], data.y.shape[0]),
                                     dtype=np.float32)
    for i in range(0, len(data.test_mask)):
        if data.test_mask[i]:
            data.y[i] = -1

    data_labels = data.y.numpy()

    #load Dataset
    graph = get_dataset.get_dataset(data_file, adjacency_matrix, data_labels, seed)

    # delete orphan points
    graph = del_outlier.del_outlier(graph)

    node_attributes = nx.get_node_attributes(graph, "attributes")
    attributes = np.array(list(node_attributes.values()))
    node_labels = nx.get_node_attributes(graph, "label")
    labels = np.array(list(node_labels.values()))
    labels = np.expand_dims(labels, axis=0)
    labels = np.reshape(labels, (nx.number_of_nodes(graph), 1))


    total_degree_dict_old = dict(graph.degree())
    total_degree_dict = {}
    id = 0
    for key, value in total_degree_dict_old.items():
        total_degree_dict[id] = value
        id += 1

    id_dict = {}
    id_dict_oldtonew = {}
    for new, old in enumerate(total_degree_dict_old):
        id_dict[new] = old
        id_dict_oldtonew[old] = new

    indices = []
    for index in total_degree_dict_old:
        indices.append(index)
    indices = np.expand_dims(indices, axis=0)
    indices = np.reshape(indices, (nx.number_of_nodes(graph), 1))
    data = np.concatenate((indices, attributes, labels), axis=1)
    data = add_id(data)

    C.append([data, total_degree_dict])


    # coarse division of granules
    new_C = initial_splite(C, graph, id_dict, id_dict_oldtonew, labels, total_degree_dict)


    target = 1
    while len(new_C) > total_balls_num:
        cut_pos = 0
        new_C.sort(key=lambda x: len(x[0]))
        for i in range(0, len(new_C)):
            if len(new_C[i][0]) == target+1:
                cut_pos = i
                break
        target += 1
        new_C = new_C[cut_pos:]

    new_C = add_purity(new_C)


    #binary division of granules
    new_C = split_ball_purity(graph, id_dict, new_C, total_degree_dict, total_balls_num)

    if len(new_C) < total_balls_num:
        new_C = split_ball_further(graph, id_dict, new_C, total_degree_dict, total_balls_num)

    new_C = purification(new_C)

    GB_features = []
    for GB in new_C:
        data = np.array(GB[0])
        slice = data[:, 1:-2]
        feature = slice.mean(axis=0)
        GB_features.append(feature)


    #build granules
    GB_graph = new_graph(new_C, graph)

    new_f = {}
    gb_labels = []
    for GB in new_C:
        gb_labels.append(GB[-1])
    new_f['gb_labels'] = np.array(gb_labels)
    C_adj = sp.coo_matrix(nx.to_numpy_array(GB_graph))
    C_adj = np.vstack((C_adj.row, C_adj.col))
    new_f['adj'] = C_adj
    new_f['gb_features'] = np.array(GB_features)

    return new_f





