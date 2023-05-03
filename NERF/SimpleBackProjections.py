# this is the same code you guys wrote this morning but in PyTorch and uses the extrinsics as well
# also its seperated from all the other code in the other repository
# also we don't need multiple libraries
import torch
import numpy as np
import cv2
from load_blender import load_blender_data, load_blender_data_test_depth

#sorry but its an absolute link its too late for me to debug this :=(
images, depths, poses, _, hwf, _ = load_blender_data_test_depth("/Users/neilhazra/NerfFinalProject/NERF/data/drums")
H, W, focal = hwf
intrinsics = torch.Tensor([[focal,0,W/2],[0,focal,H/2],[0,0,1]])
homogenous_pixels_x, homogenous_pixels_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
ones = torch.ones((W,H))
homogenous_pixel_coordinates = torch.stack([homogenous_pixels_x, homogenous_pixels_y, ones], dim = -1).float()

print(images.shape, depths.shape, poses.shape, intrinsics)
demo = images[:5,...]
demo_depth = depths[:5,...]

#800 800 3 , 800 800 1, 800 800 3, 3x3
def convert_to_camera_coord(homogenous_pixels, depth, intrinsics):
    # 800 800 3
    image_coord = homogenous_pixels * depth # something really wrong with scale probably
    camera_coord = torch.linalg.inv(intrinsics).view(1,1,3,3) @ image_coord.view(W, H, 3, 1)
    return camera_coord



camera_coord = convert_to_camera_coord(homogenous_pixel_coordinates, torch.Tensor(demo_depth[0]).unsqueeze(-1), intrinsics).detach().cpu().numpy()

print(torch.Tensor(demo_depth[0]).mean())

plotting = camera_coord.reshape(-1,3)
colors = demo[0].reshape(-1,4)

mask = (torch.Tensor(demo_depth[0]) != 0).flatten()
print(plotting[mask,:].shape)
import matplotlib.pyplot as plt
ax = plt.axes(projection ="3d")
ax.scatter3D(plotting[mask,0], plotting[mask,1], plotting[mask,2], color = colors[mask,:])
plt.show()
