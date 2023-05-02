from load_data import load_llff_data
import numpy as np

# Example code for the 20 fern images in data/llff/fern 

######################### LOAD LLFF DATA ###############################

images, poses, bds, render_poses, i_test = load_llff_data("./data/llff/fern", 8,
                                                          recenter=True, bd_factor=.75,
                                                          spherify=False)

hwf = poses[0, :3, -1] # image height, image width, focal length (in pixels)

# cast intrinsics to the right type
H, W, focal = hwf
H, W = int(H), int(W)
hwf = [H, W, focal]
# poses = poses[:,:3,:4]

print("loaded llff", images.shape, render_poses.shape, hwf, "./data/llff/fern")

########################## EXTRINSICS ##################################

# camera extrinsics for each of the 20 images
extrinsics = poses[:, :12].reshape(-1, 3, 4)
# print(extrinsics)

######################## INTRINSIC MATRIX ##############################

intrinsics = np.array([[focal,0,W/2],[0,focal,H/2],[0,0,1]])