# this is the same code you guys wrote this morning but in PyTorch and uses the extrinsics as well
# also its seperated from all the other code in the other repository
# also we don't need multiple libraries
import torch
import numpy as np
from load_blender import load_blender_data_test_depth
import matplotlib.pyplot as plt

for depth_scale in np.linspace(0.1, 0.2, 100):
    #sorry but its an absolute link its too late for me to debug this :=(
    images, depths, poses, _, hwf, _ = load_blender_data_test_depth("/Users/neilhazra/NerfFinalProject/NERF/data/drums", depth_scale=depth_scale)
    H, W, focal = hwf
    intrinsics = torch.Tensor([[focal,0,W/2],[0,focal,H/2],[0,0,1]])
    homogenous_pixels_x, homogenous_pixels_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    ones = torch.ones((W,H))
    homogenous_pixel_coordinates = torch.stack([homogenous_pixels_x, homogenous_pixels_y, ones], dim = -1).float()

    print(images.shape, depths.shape, poses.shape, intrinsics)
    demo = images[:,...]
    demo_depth = depths[:,...]
    demo_poses = poses[:, ...]


    #800 800 3 , 800 800 1, 800 800 3, 3x3
    def convert_to_camera_coord(homogenous_pixels, depth, intrinsics):
        # 800 800 3
        image_coord = homogenous_pixels * depth # something really wrong with scale probably
        camera_coord = torch.linalg.inv(intrinsics).view(1,1,3,3) @ image_coord.view(W, H, 3, 1)
        # i think there just might be one more step to get to OpenGL camera coords
        # yup we have to flip y and z
        camera_coord = torch.diag(torch.Tensor([1,-1,-1])).view(1,1,3,3) @ camera_coord
        return camera_coord

    def convert_camra_to_world(camera_coord, c2w_extrinsics):
        homogenous_camera_coord = torch.cat([camera_coord, torch.ones(W,H,1,1)], dim=-2)    
        return (c2w_extrinsics[None, None, :, :] @ homogenous_camera_coord)[..., :3,0]

    all_points = []
    all_colors = []
    for i in [20, 30, 40, 50]:
        camera_coord = convert_to_camera_coord(homogenous_pixel_coordinates, torch.Tensor(demo_depth[i]).unsqueeze(-1), intrinsics)
        world_coord = convert_camra_to_world(camera_coord, torch.Tensor(demo_poses[i]))
        world_coord = world_coord.reshape(-1,3)
        colors = demo[i].reshape(-1,4)
        mask = (torch.Tensor(demo_depth[i]) != 0).flatten()
        all_points.append(world_coord[mask,:])
        all_colors.append(colors[mask,:])

    all_points = torch.cat(all_points, dim = 0)
    all_colors = np.concatenate(all_colors, axis = 0)
    mask = np.random.choice(len(all_points), 40000)
    plotting = all_points[mask]
    colors = all_colors[mask]
    plt.figure()
    ax = plt.axes(projection ="3d")
    ax.scatter3D(plotting[:,0], plotting[:,1], plotting[:,2], s = 1, color = colors[:,:])
    plt.savefig('demo_' + str(depth_scale) + '_.png')
