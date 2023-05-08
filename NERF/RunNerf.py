import torch
import numpy as np
from load_blender import load_blender_data
from NERFFeedForwardModel import training_forward_pass, NerfModel, loss
from DataLoader import NerfDataset
import logging
import torch.optim
import os
import wandb
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

logger = logging.getLogger(__name__)


dataset = NerfDataset()
model = NerfModel().cuda()
# COMMENT/UNCOMMENT THIS LINE WHEN CONTINUING RUNS
# model.load_state_dict(torch.load("./model_256/nerf_model_40000_.model"))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

wandb.init(
    # set the wandb project where this run will be logged
    project="nerf",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "iters": 100000,
    "batch_size": 131072,
    "encoding":"positionally encoded, continued",
    "model_size":256,
    "L":10,
    }
)
num_iters = 5000

for i in range(num_iters):
    origins, ray_directions, colors = dataset.get_batch(np.random.choice(len(dataset), 32768*4))
    rendered_colors = training_forward_pass(model, origins.cuda(), ray_directions.cuda(), 2, 6, 100)
    colors = colors[:,:3]
    loss_val = loss(rendered_colors, colors.cuda())
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    print('Iteration: ', i, 'loss: ', loss_val.detach().cpu().numpy())
    
    if i % 10 == 0:
        log = f"{i} / {num_iters}"
        log += f" | loss: {loss_val.detach().cpu().numpy()}"
        logger.info(log)

        wandb.log({"loss": loss_val.detach().cpu().numpy()}) 
    
    if i % 100 == 0:
        with open('./model_256_incremental/' + 'nerf_model_' + str(i) + '_.model', 'wb') as f:
            torch.save(model.state_dict(), f)
# wandb sync /home/saumyam/NerfFinalProject/NERF/wandb/offline-run-20230504_2201
