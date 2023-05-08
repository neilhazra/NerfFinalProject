import torch
import numpy as np
from ObjectNERF import ObjectNerf
from DataLoader import NerfDataset
import torch.optim
import logging
import wandb


logger = logging.getLogger(__name__)

dataset = NerfDataset(data_root_dir='./data/drums')
model = ObjectNerf().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_iters = 100000

wandb.init(
    # set the wandb project where this run will be logged
    project="nerf",
    
    name=f"N=3, object-nerf", 
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "iters": 100000,
    "batch_size": 65536*2,
    "num_integration_points": 64,
    "mini_nerfs": 3,
    "encoding":"object nerf",
    "model_size":256,
    "L":5,
    }
)

for i in range(num_iters):
    origins, ray_directions, colors = dataset.get_batch(np.random.choice(len(dataset), 65536*2))
    rendered_colors, splits = model(origins.cuda(), ray_directions.cuda(), 2, 6, 64)
    colors = colors[:,:3]
    loss_val, c_loss, e_loss, d_loss = ObjectNerf.full_loss(rendered_colors, colors.cuda(), splits)
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    if i % 10 == 0:
        wandb.log({"loss": loss_val.item(), "color loss": c_loss.item(), "entropy loss":e_loss.item(), "model distribution": d_loss.item()}) 
    
    
    print('Iteration: ', i, 'loss: ', loss_val.item(), 
          'color loss', c_loss.item(),
          'entropy loss', e_loss.item(),
          'model distribtuion loss', d_loss.item())
    
    if i % 5000 == 0:
        with open('./model_object_N3/' + 'nerf_model_' + str(i) + '_.model', 'wb') as f:
            torch.save(model.state_dict(), f)

