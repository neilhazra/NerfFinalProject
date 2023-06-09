import torch
import numpy as np
from load_blender import load_blender_data_test_depth
from NERFFeedForwardModel import training_forward_pass, NerfModel, loss
from DataLoader import NerfDataset, pixel_intrinsics_extrinsics_to_ray
import logging
import torch.optim
import os
import wandb
import cv2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


model = NerfModel().cuda()
model.load_state_dict(torch.load("./model_50k_positional(L5)/nerf_model_60000_.model"))

images, _, poses, _, [H, W, focal], _ = load_blender_data_test_depth("./data/drums")

intrinsics = torch.Tensor([[focal,0,W/2],[0,focal,H/2],[0,0,1]])
homogenous_pixels_x, homogenous_pixels_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
ones = torch.ones((W,H))
homogenous_pixel_coordinates = torch.stack([homogenous_pixels_x, homogenous_pixels_y, ones], dim = -1).float()
print(images.shape, poses.shape, intrinsics)
print('generating rays')

losses = torch.zeros(200)
with torch.no_grad():
    #for i in range(200):
    for i in [199]:
        print('processing image', i)
        translation, world_direction = pixel_intrinsics_extrinsics_to_ray(homogenous_pixel_coordinates, intrinsics, torch.Tensor(poses[i][:3,:])) 
        rendered_color_splits = []
        for split in range(4):
            split_translation = translation.expand_as(world_direction).reshape(-1,3)[split*160000: (split+1)*160000]
            split_world_direction = world_direction.reshape(-1,3)[split*160000: (split+1)*160000]
            rendered_colors = training_forward_pass(model, split_translation.cuda(), split_world_direction.cuda(), 2, 6, 100)
            rendered_color_splits.append(rendered_colors)
        
        # SUM OVER RGB, AVERAGE OVER NUM PIXELS
        rendered_colors = torch.cat(rendered_color_splits, dim=0).reshape(800,800,3)    

        # COMPUTING AVERAGE LOSS
        # iter_loss = torch.sum((rendered_colors.cpu() - images[i][:,:,:3])**2)
        # avg_loss = iter_loss / (800*800)
        # losses[i] = avg_loss
        # print(avg_loss)
        
        # DISPLAYING IMAGE
        print('we have an image', rendered_colors.detach().cpu().numpy().max())
        print('real image', images[i].mean())
        cv2.imwrite('test_199_N1_60k_L5.png', np.flip((rendered_colors.detach().cpu().numpy()*255).astype(int), axis=-1))
        cv2.imwrite('actual.png', np.flip((images[i][...,:3]*255).astype(int), axis=-1))
        
    
    print("Average loss:", losses.mean())
    
    