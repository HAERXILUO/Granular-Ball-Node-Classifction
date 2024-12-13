import os
import argparse
from gb_division import gb_division
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from tools.split_indices import split_indices
from split_test import get_planetoid_dataset
from split_test import get_coauthor_dataset
from models.APPNP import appnp_c
from models.GAT import gat
import copy


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, args.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x,training= self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def main():
    #parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--split_data', type=str, default="normal")
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--models', type=str, default='GCN')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--ball_r', type=float, default=0.3)
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--heads', type=int, default=8)

    args = parser.parse_args()
    path = "params/"
    if not os.path.isdir(path):
        os.mkdir(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #dataset selection
    if args.dataset == "physics" or args.dataset == "cs":
        dataset = get_coauthor_dataset(args.dataset, split=args.split_data)
    else:
        dataset = get_planetoid_dataset(args.dataset, split=args.split_data)

    data = dataset
    fun_data = copy.deepcopy(data)

    #GB division
    new_data = gb_division(fun_data, args)

    data = data.to(device)
    args.num_classes = len(set(np.array(data.y)))
    args.gb_labels = new_data['gb_labels']
    features = torch.from_numpy(new_data['gb_features'])
    args.num_features = len(features[0])


    if args.models == 'GCN':
        model = GCN(args).to(device)
    elif args.models == 'APPNP':
        model = appnp_c(args).to(device)
    elif args.models == 'GAT':
        model = gat(args).to(device)
    else:
        print("worng models!")
        exit(0)

    all_acc = []

    #training
    for i in range(args.runs):
        train_list, val_list = split_indices(list(range(len(new_data['gb_labels']))), 20, ways="random")
        train_index = torch.tensor(train_list).to(device)
        val_index = torch.tensor(val_list).to(device)
        new_data['train_mask'] = torch.zeros(len(new_data['gb_labels']), dtype=torch.bool)
        new_data['val_mask'] = torch.zeros(len(new_data['gb_labels']), dtype=torch.bool)
        new_data['train_mask'][train_index] = True
        new_data['val_mask'][val_index] = True
        new_features = features.to(torch.float)
        new_adj = torch.from_numpy(new_data['adj']).to(torch.int64)
        new_labels = torch.from_numpy(new_data['gb_labels']).to(torch.int64)
        new_features = new_features.to(device)
        new_adj = new_adj.to(device)
        new_labels = new_labels.to(device)
        new_data['train_mask'] = new_data['train_mask'].to(device)
        new_data['val_mask'] = new_data['val_mask'].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val_loss = float('inf')
        val_loss_history = []

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            out = model(new_features, new_adj)
            loss = F.nll_loss(out[new_data['train_mask']], new_labels[new_data['train_mask']])
            loss.backward()
            optimizer.step()
            model.eval()
            pred = model(new_features, new_adj)
            val_loss = F.nll_loss(pred[new_data['val_mask']], new_labels[new_data['val_mask']]).item()
            if val_loss < best_val_loss and epoch > args.epochs // 2:
                best_val_loss = val_loss
                torch.save(model.state_dict(), path + 'checkpoint-best-acc.pkl')
            val_loss_history.append(val_loss)
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
        model.load_state_dict(torch.load(path + 'checkpoint-best-acc.pkl'))
        model.eval()
        pred = model(data.x, data.edge_index).max(1)[1]
        test_acc = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()) / int(data.test_mask.sum())
        print(test_acc)
        all_acc.append(test_acc)

    print('ave_acc: {:.4f}'.format(np.mean(all_acc)), '+/- {:.4f}'.format(np.std(all_acc)))

if __name__ == '__main__':

    main()

