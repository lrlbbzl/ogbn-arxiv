import torch
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from copy import deepcopy

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='ogbn-dataset')
data = dataset[0]
split_idx = dataset.get_idx_split()
test_idx, edge_idx = split_idx['test'], data.edge_index # type of element in edge_idx is torch.Tensor 
test_idx_set = set(test_idx.numpy())

# Remove the edges connected to nodes in test set
train_edge = [[], []]
for i in range(edge_idx.shape[1]):
    if edge_idx[0][i].item() not in test_idx_set and edge_idx[1][i].item() not in test_idx_set:
        train_edge[0].append(edge_idx[0][i].item())
        train_edge[1].append(edge_idx[1][i].item())

total_data = deepcopy(data) # dataset which includes total edges
data.edge_index = torch.tensor(train_edge) # dataset which excludes the edges connected to test set

sparser = T.ToSparseTensor()
data = sparser(data)
total_data = sparser(total_data)

print(data)

data.adj_t = data.adj_t.to_symmetric()
total_data.adj_t = total_data.adj_t.to_symmetric()

torch.save(data, 'ogbn-dataset/data.pth')
torch.save(total_data, 'ogbn-dataset/total_data.pth')
