import torch.nn as nn
import torch.nn.functional as F
import torch

class NerfModel(nn.Module):
    def __init__(self):
      super(NerfModel, self).__init__()
      self.linear1 = nn.Linear(5, 128) # x,y,z, theta, phi
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
   return world_points_centered - extrinsics_translation[None, None, :]
















   return None