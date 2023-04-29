import torch.nn as nn
import torch.nn.functional as F
import torch

class NerfModel(nn.Module):
    def __init__(self):
      super(NerfModel, self).__init__()
      self.linear1 = nn.Linear(6, 128) # x,y,z, normalized_direction_vector
      self.linear2 = nn.Linear(128, 128)
      self.linear3 = nn.Linear(128, 128)
      self.linear4 = nn.Linear(128, 4) # r,g,b, sigma

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.linear1(x)
      x = F.relu(x)
      x = self.linear2(x)
      x = F.relu(x)
      x = self.linear3(x)
      x = F.relu(x)
      x = self.linear4(x)
      # Apply softmax to x
      output = x #F.sigmoid(x, dim=1)
      return output

# get points along a ray
# make sure there is a point in every section of the ray
# add some random noise to make sure we don't sample the ray at the same point during
# every iteration
def get_scalar_array(scene_start, scene_end, num_integration_points):
   stratified_points = torch.linspace(scene_start, scene_end, num_integration_points)
   random_offsets = torch.rand(num_integration_points) * (scene_end - scene_start) / (num_integration_points - 1)
   return stratified_points + random_offsets

# this function takes in batch pixel location homogenous coordinates 
# batch camera intrinsics and batch camera extrinsics
# oh and also scalar array is a very simple array of scalars i.e. [1,2,3,4,5] that 
# corresponds to how fine the integration will be
# It returns a array of points sampled along the 3d world line
def pixel_intrinsics_extrinsics_to_ray(pixel_homogenous, intrinsics, extrinsics, scalar_array):  
   pixel_homogenous_shape = pixel_homogenous.shape # this should be (batch_size, 3)
   intrinsics_shape = intrinsics.shape # (batch_size, 3, 3)
   extrinsics_shape = extrinsics.shape # (batch_size, 3, 4)
   # all shapes of the inputs
   print('pixel shape, intrinsics shape, extrinsics shape', pixel_homogenous_shape, intrinsics_shape, extrinsics_shape)
   extrinsics_rotation = extrinsics[..., :3]
   extrinsics_translation = extrinsics[...,4]
   # batch matrix multiplication, we want camera_coordinate to be of shape 
   # (batch x 3 x 3) * (batch x 3 x 1) = batch x 3 x 1
   camera_coordinate = torch.linalg.inv(intrinsics) @ pixel_homogenous
   # now we get the world coordinate direction by using the inverse extrinsics rotation matrix
   world_direction = torch.linalg.inv(extrinsics_rotation) @ camera_coordinate
   # normalized world direction  (batch_size, 3)
   normalized_world_direction = (world_direction / torch.linalg.norm(world_direction, dim = -2, keepdim = True)).squeeze(-1)
   # batch_size, num_integration_points, 3
   world_points_centered = normalized_world_direction[:,None, :] * scalar_array[None, :, None]
   return world_points_centered - extrinsics_translation[None, None, :], normalized_world_direction # i think this is subtraction, but not sure


def evaluate_nerf_along_ray(nerf_model, world_points, normalized_world_direction):
  # should print batch_size, num_integration_points, 3 and (batch_size, 3)
  print(world_points.shape, normalized_world_direction.shape)
  # next we need to expand normalized_world_direction to be the same size and then concatenate
  batch_size = world_points.shape[0]
  num_integration_points = world_points.shape[1]
  normalized_world_direction = normalized_world_direction[:,None,:].expand_as(world_points)
  # should be on shape batch_size, num_integration_points, 6
  nn_in = torch.cat(world_points, normalized_world_direction, dim = -1)
  return nerf_model(nn_in.view(-1, 6)).view(batch_size, num_integration_points, 4)
   

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
   
# takes in nerf model, batched homogenous pixels, batched intrinsics, batched extrinsics, scene start, scene end, num points for integration
def full_forward_pass(model, pixel_homogenous, intrinsics, extrinsics, scene_start, scene_end, num_integration_points):
  scalar_array = get_scalar_array(scene_start, scene_end, num_integration_points)
  ray, direction = pixel_intrinsics_extrinsics_to_ray(pixel_homogenous, intrinsics, extrinsics, scalar_array)
  nerf_eval = evaluate_nerf_along_ray(model, ray, direction)
  rendered_color = integrate(nerf_eval, scalar_array)
  return rendered_color 











