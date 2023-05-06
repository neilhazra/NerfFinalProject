# my own design and idea inspired by BlockNerf and Recursive NERF
# https://waymo.com/research/block-nerf/
# and https://arxiv.org/pdf/2105.09103.pdf

import torch.nn as nn
import torch.nn.functional as F
import torch

L = 10 # positional encodings
N = 8 # number of mini-nerf models

class ModelSplitter(nn.Module):
  #maybe make this more complex
  def __init__(self):
    super(ModelSplitter, self).__init__()
    self.linear1 = nn.Linear(3, N) # x,y,z
  def forward(self, x):
    return F.softmax(self.linear1(x), dim=-1)

class MiniNerfModel(nn.Module):
    def __init__(self):
      super(MiniNerfModel, self).__init__()
      # if positionally encoded, 6 * L + 3 x,y,z original + direction
      self.linear1 = nn.Linear(6*L + 6, 128) # x,y,z, normalized_direction_vector
      self.linear2 = nn.Linear(128, 128)
      self.linear3 = nn.Linear(128, 4) # r,g,b, sigma

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.linear1(x)
      x = F.relu(x)
      x = self.linear2(x)
      x = F.relu(x)
      x = self.linear3(x)
      output = x 
      return output

class ObjectNerf(nn.Module):
    def __init__(self):
      super(ObjectNerf, self).__init__()
      self.mini_nerfs = [MiniNerfModel() for i in range(N)] # generate all the mini-nerfs
      self.splitter = ModelSplitter()

    def forward(self, ray_origin, ray_direction, scene_start, scene_end, num_integration_points):
      scalar_array = get_scalar_array(scene_start, scene_end, num_integration_points).to(ray_origin) #map to same device automatically
      ray = ray_direction[:,None, :] * scalar_array[None, :, None] + ray_origin[:,None,:]
      evals = [evaluate_nerf_along_ray(model, ray, ray_direction) for model in self.mini_nerfs]
      splits = self.splitter(ray)
      evals = torch.stack(evals, dim=-1)
      nerf_eval = (evals * splits[:,:,None, :]).sum(dim = -1)
      rendered_color = integrate(nerf_eval, scalar_array)
      print(rendered_color.shape)
      return rendered_color, splits
    
    #the goal is that during eval we don't want to evaluate all the mini nerfs
    # for every point there should be one mininerf that is of importance
    # thus we want the splitting distribution to become the degenerate distribution as the
    # training progresses
    def entropy_loss(splits):
      return (-(splits * torch.log(splits)).sum(dim = -1)).mean()
    
    def color_loss(color_x, color_y):
      return torch.sum((color_x - color_y)**2, dim=-1).mean()

    def full_loss(rendered_color, actual_color, splits, l=1.0):
       c_loss = ObjectNerf.color_loss(rendered_color, actual_color)
       e_loss = ObjectNerf.entropy_loss(splits)
       return c_loss + e_loss*l
       

# get points along a ray
# make sure there is a point in every section of the ray
# add some random noise to make sure we don't sample the ray at the same point during
# every iteration
@torch.no_grad()
def get_scalar_array(scene_start, scene_end, num_integration_points):
   stratified_points = torch.linspace(scene_start, scene_end, num_integration_points)
   random_offsets = torch.rand(num_integration_points) * (scene_end - scene_start) / (num_integration_points - 1)
   return stratified_points + random_offsets


def evaluate_nerf_along_ray(nerf_model, world_points, normalized_world_direction, is_pos_encoded=True):
  # should print batch_size, num_integration_points, 3 and (batch_size, 3)
  # next we need to expand normalized_world_direction to be the same size and then concatenate
  batch_size = world_points.shape[0]
  num_integration_points = world_points.shape[1]
  normalized_world_direction = normalized_world_direction[:,None,:].expand_as(world_points)
  
  # this is of shape [batch_size, num_integration_points, 3 (XYZ COORDs)]
  # you want to convert this to something like
  # batchsize, num_integration_points
  # cos(x), cos(2x), cos(3x), cos(y), cos(2y), cos(3y) same thing for z
  # same thing for sin
  # and then yyou wanna concantenate all of them and put into nn
  # basically after we just change input layer of nn
  # should be on shape batch_size, num_integration_points, 6
  if is_pos_encoded:
    pos_encodings = [world_points]
    for i in range(L):
        pos_encodings.append(torch.cos((2**i) * torch.pi* world_points))
        pos_encodings.append(torch.sin((2**i) * torch.pi* world_points))
    world_points = torch.cat(pos_encodings, dim=2)  
  nn_in = torch.cat([world_points, normalized_world_direction], dim = -1)
  return nerf_model(nn_in.view(-1, 6*L + 6)).view(batch_size, num_integration_points, 4)
  
# nerf eval is of size batch_size, num_integration_points, 4
# scalar_array is of shape num_integration_points
def integrate(nerf_eval, scalar_array):
  color =  nerf_eval[..., :3] # batch_size, num_integration_points, 3
  sigma = nerf_eval[..., 3] # batch_size, num_integration_points
  diffs = torch.diff(scalar_array) # num_integration_points-1
  # see page 6 in https://arxiv.org/pdf/2003.08934.pdf
  T = torch.exp(torch.cumsum(sigma[...,:-1] * diffs[None, :], dim = -1)) #batch_size, num_integration_points-1
  mid_term = 1 - torch.exp(-sigma[...,:-1] * diffs[None, :]) #batch_size, num_integration_points-1
  return ((T * mid_term)[..., None] * color[:,:-1,:]).sum(dim=-2) # batch_size, 3





if __name__ == '__main__':
    from DataLoader import NerfDataset
    import numpy as np
    dataset = NerfDataset(data_root_dir='/Users/neilhazra/NerfFinalProject/NERF/data/drums')
    model = ObjectNerf()
    origins, ray_directions, colors = dataset.get_batch(np.random.choice(len(dataset), 44))
    rendered_colors = model(origins, ray_directions, 2, 6, 100)
    
    
