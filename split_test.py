import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.datasets import Coauthor
from torch_geometric.data import Data


def get_planetoid_dataset(name, normalize_features=False, transform=None, split="normal"):
    if split == 'complete':
        dataset = Planetoid(root='dataset', name=name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    elif split == "normal":
        dataset = Planetoid(root='dataset', name=name)
        num_nodes = dataset[0].num_nodes
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)
        dataset[0].train_mask.fill_(False)
        dataset[0].val_mask.fill_(False)
        dataset[0].test_mask.fill_(False)
        dataset[0].train_mask[:num_train] = True
        dataset[0].val_mask[num_train:num_train + num_val] = True
        dataset[0].test_mask[num_train + num_val:] = True

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset[0]


def get_coauthor_dataset(name, normalize_features=False, transform=None, split="normal"):
    if split == 'complete':
        dataset = Coauthor(root='dataset/' + name, name=name)
        new_dataset = Data(edge_index=dataset[0].edge_index, x=dataset[0].x, y=dataset[0].y)
        train_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        train_mask[:dataset[0].num_nodes - 1000] = 1
        val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        test_mask[dataset[0].num_nodes - 500:] = 1
        new_dataset.train_mask = train_mask
        new_dataset.val_mask = val_mask
        new_dataset.test_mask = test_mask
    elif split == "normal":
        dataset = Coauthor(root='dataset/' + name, name=name)
        new_dataset = Data(edge_index=dataset[0].edge_index, x=dataset[0].x, y=dataset[0].y)
        train_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(dataset[0].num_nodes, dtype=torch.bool)
        num_nodes = dataset[0].num_nodes
        num_train = int(num_nodes * 0.6)
        num_val = int(num_nodes * 0.2)
        train_mask[:num_train] = True
        val_mask[num_train:num_train + num_val] = True
        test_mask[num_train + num_val:] = True
        new_dataset.train_mask = train_mask
        new_dataset.val_mask = val_mask
        new_dataset.test_mask = test_mask

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return new_dataset


