import torch
import numpy as np
from load_blender import load_blender_data
from NERFFeedForwardModel import training_forward_pass, NerfModel, loss
from DataLoader import NerfDataset
import torch.optim
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

dataset = NerfDataset()
model = NerfModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_iters = 10000

for i in range(num_iters):
    origins, ray_directions, colors = dataset.get_batch(np.random.choice(len(dataset), 65536))
    rendered_colors = training_forward_pass(model, origins.cuda(), ray_directions.cuda(), 2, 6, 100)
    colors = colors[:,:3]
    loss_val = loss(rendered_colors, colors.cuda())
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    print(loss_val.detach().cpu().numpy())
    if i % 1000 == 0:
        with open('nerf_model_' + str(i) + '_.model', 'wb') as f:
            torch.save(model.state_dict, f)
