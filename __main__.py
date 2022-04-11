import argparse
from configparser import NoOptionError
from attr import validate
import torch
import torch.nn.functional as F
import numpy as np
from ogb.nodeproppred import Evaluator
from model import GCN, GraphSAGE, GAT, JKNet, GCNRes
from ogb.nodeproppred import PygNodePropPredDataset
from logger import Logger
from attack import flag

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    output = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(output, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

def train_with_FLAG(model, data, train_idx, optimizer, args, device):
    loss = flag(model, data, train_idx, args, optimizer, device)
    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    output =  model(data.x, data.adj_t)
    pred = output.argmax(dim=-1, keepdim=True)
    train_accuracy = evaluator.eval({'y_true' : data.y[split_idx['train']], 'y_pred' : pred[split_idx['train']]})['acc']
    validate_accuracy = evaluator.eval({'y_true' : data.y[split_idx['valid']], 'y_pred' : pred[split_idx['valid']]})['acc']
    test_accuracy = evaluator.eval({'y_true' : data.y[split_idx['test']], 'y_pred' : pred[split_idx['test']]})['acc']
    return train_accuracy, validate_accuracy, test_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ogbn-arxiv')
    parser.add_argument('--model', type=str, default='GCNRes')
    parser.add_argument('--hidden_channel', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--att_dropout', type=float, default=0.2)
    parser.add_argument('--mode', type=str, default='max')
    parser.add_argument('--m', type=int, default=3)
    parser.add_argument('--attack', type=bool, default=True)
    parser.add_argument('--ascent_step_size', type=float, default=1e-3)
    
    args = parser.parse_args()
    evaluator = Evaluator(name='ogbn-arxiv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cuda' if torch.cuda.is_available() else 'cpu')
    # load data
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=None)
    split_idx = dataset.get_idx_split()
    data, total_data = torch.load('ogbn-dataset/data.pth'), torch.load('ogbn-dataset/total_data.pth')
    model_map = dict({'GCN' : GCN(data.num_features, args.hidden_channel, dataset.num_classes, args.num_layers, args.dropout_rate).to(device),
                    'GraphSAGE' : GraphSAGE(data.num_features, args.hidden_channel, dataset.num_classes, args.num_layers, args.dropout_rate).to(device),
                    'GAT' : GAT(data.num_features, args.hidden_channel, dataset.num_classes, args.num_layers, args.dropout_rate, args.heads, args.att_dropout).to(device),
                    'JKNet' : JKNet(data.num_features, args.hidden_channel, dataset.num_classes, args.num_layers, args.dropout_rate, args.mode).to(device),
                    'GCNRes' : GCNRes(data.num_features, args.hidden_channel, dataset.num_classes, args.num_layers, args.dropout_rate).to(device)
                    })
    model = model_map[args.model]
    data.to(device)
    total_data.to(device)
    logger = Logger(args.runs)
    # train
    for run in range(args.runs):
        model.reset_params()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        for epoch in range(1, 1 + args.epochs):
            if not args.attack:
                loss = train(model, data, split_idx['train'], optimizer)
            else:
                loss = train_with_FLAG(model, data, split_idx['train'], optimizer, args, device)
            train_acc, validate_acc, test_acc = test(model, total_data, split_idx, evaluator)
            logger.update(run, [train_acc, validate_acc, test_acc])
        logger.print(run, args)
    logger.print(None, args)
    



