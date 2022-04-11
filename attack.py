import torch
import torch.nn.functional as F
import numpy as np

def flag(model, data, train_idx, args, optimizer, device):
    model.train()
    optimizer.zero_grad()

    y = data.y.squeeze()[train_idx]
    perturbation = torch.FloatTensor(data.x.shape).uniform_(-args.ascent_step_size, args.ascent_step_size).to(device)
    perturbation.requires_grad_() # the perturbation need to save the grad when backprop
    output = model(data.x + perturbation, data.adj_t)[train_idx]
    loss = F.nll_loss(output, y)
    loss /= args.m

    for i in range(args.m - 1):
        loss.backward()
        perturbation.data = perturbation.detach() + args.ascent_step_size * torch.sign(perturbation.grad.detach())
        perturbation.grad[:] = 0
        output = model(data.x + perturbation, data.adj_t)[train_idx]
        loss = F.nll_loss(output, y)
        loss /= args.m
    
    loss.backward()
    optimizer.step()
    return loss
