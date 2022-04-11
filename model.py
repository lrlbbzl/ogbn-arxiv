import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP, JumpingKnowledge
from transformers import BatchEncoding
from copy import deepcopy

class GCN(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, dropout_rate):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channel, hidden_channel))
        for i in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channel, hidden_channel))
        self.convs.append(GCNConv(hidden_channel, out_channel))
        self.batchNorm = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.batchNorm.append(torch.nn.BatchNorm1d(hidden_channel))
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def reset_params(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batchNorm:
            bn.reset_parameters()
    
    def forward(self, x, adj_m):
        for i, conv in enumerate(self.convs[ : -1]):
            x = conv(x, adj_m)
            x = self.batchNorm[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.convs[-1](x, adj_m)
        return F.log_softmax(x, dim=-1)

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, dropout_rate):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channel, hidden_channel))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channel, hidden_channel))
        self.convs.append(SAGEConv(hidden_channel, out_channel))
        self.batchNorm = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.batchNorm.append(torch.nn.BatchNorm1d(hidden_channel))
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def reset_params(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batchNorm:
            bn.reset_parameters()
    
    def forward(self, x, adj_m):
        for i, conv in enumerate(self.convs[ : -1]):
            x = conv(x, adj_m)
            x = self.batchNorm[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.convs[-1](x, adj_m)
        return F.log_softmax(x, dim=-1)

class GAT(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, dropout_rate, heads, att_dropout):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels=in_channel, out_channels=hidden_channel, heads=heads, dropout=att_dropout))
        for i in range(num_layers - 2):
            self.convs.append(GATConv(in_channels=hidden_channel * heads, out_channels=hidden_channel, heads=heads, dropout=att_dropout))
        self.convs.append(GATConv(in_channels=hidden_channel * heads, out_channels=out_channel, heads=heads, dropout=att_dropout, concat=False))
        self.batchNorm = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.batchNorm.append(torch.nn.BatchNorm1d(hidden_channel * heads))
        self.activation = torch.nn.LeakyReLU(0.1) # maybe it can be another ReLU...
        self.dropout = torch.nn.Dropout(p=dropout_rate)
    
    def reset_params(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batchNorm:
            bn.reset_parameters()
    
    def forward(self, x, adj_m):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_m)
            x = self.batchNorm[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.convs[-1](x, adj_m)
        return F.log_softmax(x, dim=-1)

class JKNet(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, dropout_rate, mode):
        super(JKNet, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channel, hidden_channel))
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channel, hidden_channel))
        self.batchNorm = torch.nn.ModuleList()
        for i in range(num_layers):
            self.batchNorm.append(torch.nn.BatchNorm1d(hidden_channel))
        if mode == 'lstm':
            self.jump = JumpingKnowledge(mode, hidden_channel, num_layers - 1)
        else:
            self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            hidden_channel = hidden_channel * num_layers
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.output = torch.nn.Linear(hidden_channel, out_channel)
    
    def reset_params(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batchNorm:
            bn.reset_parameters()
    
    def forward(self, x, adj_m):
        lst = []
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_m)
            x = self.batchNorm[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            lst.append(x)
        x = self.jump(lst)
        x = self.output(x)
        return F.log_softmax(x, dim=-1)

class GCNRes(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, dropout_rate):
        super(GCNRes, self).__init__()
        self.input = torch.nn.Linear(in_channel, hidden_channel)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_channel, hidden_channel, ))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channel))
        self.output = torch.nn.Linear(hidden_channel, out_channel)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.weights = torch.nn.Parameter(torch.randn(len(self.convs)))

    def reset_params(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input.reset_parameters()
        self.output.reset_parameters()
        torch.nn.init.normal_(self.weights)
    
    def forward(self, x, adj_m):
        x = self.input(x)
        x_c = x

        lst = []
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_m)
            x = self.bns[i](x)
            x = self.activation(x)
            x = self.dropout(x)

            if i == 0:
                x = x + 0.2 * x_c
            else:
                x = x + 0.2 * x_c + 0.5 * lst[-1]
            lst.append(x)
        weight = torch.softmax(self.weights, dim=-1)
        for i in range(len(weight)):
            lst[i] = lst[i] * weight[i]
        x = sum(lst)
        x = self.output(x)
        return F.log_softmax(x, dim=-1)


