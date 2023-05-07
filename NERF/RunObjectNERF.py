import torch
import numpy as np
from ObjectNERF import ObjectNerf
from DataLoader import NerfDataset
import torch.optim



dataset = NerfDataset(data_root_dir='/Users/neilhazra/NerfFinalProject/NERF/data/drums')
model = ObjectNerf().cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_iters = 100000

for i in range(num_iters):
    origins, ray_directions, colors = dataset.get_batch(np.random.choice(len(dataset), 44))
    rendered_colors, splits = model(origins.cpu(), ray_directions.cpu(), 2, 6, 100)
    colors = colors[:,:3]
    loss_val, c_loss, e_loss, d_loss = ObjectNerf.full_loss(rendered_colors, colors.cpu(), splits)
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    print('Iteration: ', i, 'loss: ', loss_val.detach().cpu().numpy(), 
          'color loss', c_loss.detach().cpu().numpy(),
          'entropy loss', e_loss.detach().cpu().numpy(),
          'model distribtuion loss', d_loss.detach().cpu().numpy())

