import torch
import numpy as np
from load_blender import load_blender_data
# from load_data import recenter_poses
from NERFFeedForwardModel import training_forward_pass, NerfModel, loss


def pixel_intrinsics_extrinsics_to_ray(pixel_homogenous, intrinsics, c2w_3x4):
   print(pixel_homogenous.shape, intrinsics.shape, c2w_3x4.shape)  
   pixel_homogenous_shape = pixel_homogenous.shape # this should be (batch_size, 3)
   intrinsics_shape = intrinsics.shape # (batcƒh_size, 3, 3)
   extrinsics_shape = c2w_3x4.shape # (batch_size, 3, 4)
   # all shapes of the inputs
   print('pixel shape, intrinsics shape, extrinsics shape', pixel_homogenous_shape, intrinsics_shape, extrinsics_shape)
   extrinsics_rotation = c2w_3x4[..., :3]
   extrinsics_translation = c2w_3x4[...,3]
   # batch matrix multiplication, we want camera_coordinate to be of shape 
   # (batch x 3 x 3) * (batch x 3 x 1) = batch x 3 x 1
   camera_coordinate = torch.linalg.inv(intrinsics).view(1,1,3,3) @ pixel_homogenous[...,None]
   # this is to go to the OpenGL coordinate system where y is positive up, and z is positive backwards
   camera_coordinate = torch.diag(torch.Tensor([1,-1,-1])).view(1,1,3,3) @ camera_coordinate
   # now we get the world coordinate direction by using the inverse extrinsics rotation matrix
   # add the extra batch dimension to broadcast the matmul
   world_direction = extrinsics_rotation[None, ...] @ camera_coordinate
   # normalized world direction  (batch_size, 3)
   normalized_world_direction = (world_direction / torch.linalg.norm(world_direction, dim = -2, keepdim = True)).squeeze(-1)
   # batch_size, num_integration_points, 3
   return extrinsics_translation, normalized_world_direction


class NerfDataset():
    def __init__(self, data_root_dir = "./data/drums"):
        self.images, self.poses, _, hwf, _ = load_blender_data(data_root_dir)
        # self.poses = recenter_poses(self.poses)
        self.H, self.W, self.focal = hwf
        self.intrinsics = torch.Tensor([[self.focal,0,self.W/2],[0,self.focal,self.H/2],[0,0,1]])
        homogenous_pixels_x, homogenous_pixels_y = torch.meshgrid(torch.arange(self.W), torch.arange(self.H), indexing='xy')
        ones = torch.ones((self.W,self.H))
        self.homogenous_pixel_coordinates = torch.stack([homogenous_pixels_x, homogenous_pixels_y, ones], dim = -1).float()
        print(self.images.shape, self.poses.shape, self.intrinsics)
        print('generating rays')

        all_camera_centers = []
        all_world_directions = []
        all_colors = []

        for i in range(self.images.shape[0]):
            print('processing image', i)
            translation, world_direction = pixel_intrinsics_extrinsics_to_ray(self.homogenous_pixel_coordinates, self.intrinsics, torch.Tensor(self.poses[i][:3,:])) 
            all_camera_centers.append(translation.expand_as(world_direction).reshape(-1,3))
            all_world_directions.append(world_direction.reshape(-1,3))
            all_colors.append(torch.Tensor(self.images[i].reshape(-1,4)))

        self.origins = torch.cat(all_camera_centers, dim = 0)
        self.rays = torch.cat(all_world_directions, dim = 0)
        self.colors = torch.cat(all_colors, dim = 0)


    def __len__(self):
        return self.origins.shape[0]

    @torch.no_grad()
    def get_batch(self, batch_indices):
        return self.origins[batch_indices,...], self.rays[batch_indices,...], self.colors[batch_indices,...]

if __name__ == "__main__":
    dataset = NerfDataset()
    origins, ray_directions, colors = dataset.get_batch(np.random.choice(len(dataset), 256))
    colors = colors[:,:3]
    print(origins.shape, ray_directions.shape, colors.shape)
    model = NerfModel()
    rendered_colors = training_forward_pass(model, origins, ray_directions, 2, 6, 10)
    #print(rendered_colors.shape, colors.shape)
    print(loss(rendered_colors, colors))
