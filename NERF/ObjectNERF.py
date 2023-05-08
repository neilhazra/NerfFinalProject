# my own design and idea inspired by BlockNerf and Recursive NERF
# https://waymo.com/research/block-nerf/
# and https://arxiv.org/pdf/2105.09103.pdf
# results in faster performance since the model is smaller and split
# between multiple neural networks

import torch.nn as nn
import torch.nn.functional as F
import torch

L = 5 # positional encodings
N = 2 # number of mini-nerf models

class ModelSplitter(nn.Module):
  #maybe make this more complex
  def __init__(self):
    super(ModelSplitter, self).__init__()
    self.linear1 = nn.Linear(3, 64) # x,y,z
    self.linear2 = nn.Linear(64, N) # x,y,z
  def forward(self, x):
    x = self.linear1(x)
    x = F.relu(x)
    x = self.linear2(x)
    return F.softmax(x, dim=-1)

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
      self.mini_nerfs = nn.ModuleList([MiniNerfModel() for i in range(N)]) # generate all the mini-nerfs
      self.splitter = ModelSplitter()

    def forward(self, ray_origin, ray_direction, scene_start, scene_end, num_integration_points):
      scalar_array = get_scalar_array(scene_start, scene_end, num_integration_points).to(ray_origin) #map to same device automatically
      ray = ray_direction[:,None, :] * scalar_array[None, :, None] + ray_origin[:,None,:]
      evals = [evaluate_nerf_along_ray(model, ray, ray_direction) for model in self.mini_nerfs]
      splits = self.splitter(ray)
      evals = torch.stack(evals, dim=-1)
      nerf_eval = (evals * splits[:,:,None, :]).sum(dim = -1)
      rendered_color = integrate(nerf_eval, scalar_array)
      return rendered_color, splits
    
    def forward_eval(self, ray_origin, ray_direction, scene_start, scene_end, num_integration_points):
      scalar_array = get_scalar_array(scene_start, scene_end, num_integration_points).to(ray_origin) #map to same device automatically
      ray = ray_direction[:,None, :] * scalar_array[None, :, None] + ray_origin[:,None,:]
      evals = [evaluate_nerf_along_ray(model, ray, ray_direction) for model in self.mini_nerfs]
      splits = self.splitter(ray)
      max_idx = torch.argmax(splits, -1, keepdim=True)
      one_hot = torch.zeros(splits.shape).to(splits)
      one_hot.scatter_(-1, max_idx, 1)
      evals = torch.stack(evals, dim=-1)
      nerf_eval = (evals * one_hot[:,:,None, :]).sum(dim = -1)
      rendered_color = integrate(nerf_eval, scalar_array)
      return rendered_color, splits
    
    #the goal is that during eval we don't want to evaluate all the mini nerfs
    # for every point there should be one mininerf that is of importance
    # thus we want the splitting distribution to become the degenerate distribution as the
    # training progresses
    def entropy_loss(splits):
      return (-(splits * torch.log(splits + 0.001)).sum(dim = -1)).mean()
    
    # we really don't want all the distributions to look the same
    # hard problem since we want degenerate distributions, but we 
    # also want all networks to be sampled equally
    def degeneracy_loss(splits):
      empirical_model_distribution = splits.mean(dim=[0,1])
      return (-(empirical_model_distribution * torch.log(empirical_model_distribution)).sum())

    def color_loss(color_x, color_y):
      return torch.sum((color_x - color_y)**2, dim=-1).mean()


    # c loss should be minimized since colors should be similar
    # e loss should be minimized since each point should only correspond to one model
    # d loss should be maximized because we want all models to be equally used
    def full_loss(rendered_color, actual_color, splits, l= 1, alpha=1):
       c_loss = ObjectNerf.color_loss(rendered_color, actual_color)
       e_loss = ObjectNerf.entropy_loss(splits)
       d_loss = ObjectNerf.degeneracy_loss(splits)
       return c_loss + (e_loss*alpha - d_loss)*l, c_loss, e_loss, d_loss
       

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
    dataset = NerfDataset(data_root_dir='./data/drums')
    model = ObjectNerf()
    origins, ray_directions, colors = dataset.get_batch(np.random.choice(len(dataset), 44))
    rendered_colors = model(origins, ray_directions, 2, 6, 100)

    
