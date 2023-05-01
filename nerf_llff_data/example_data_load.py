from load_data import load_llff_data
import numpy as np

# Example code for the 20 fern images in data/llff/fern 


######################### LOAD LLFF DATA ###############################

images, poses, bds, render_poses, i_test = load_llff_data("./data/llff/fern", 8,
                                                          recenter=True, bd_factor=.75,
                                                          spherify=True)

########################## INTRINSICS ##################################

hwf = poses[0, :3, -1] # image height, image width, focal length (in pixels)

# cast intrinsics to the right type
H, W, focal = hwf
H, W = int(H), int(W)
hwf = [H, W, focal]

pixel_offset = np.array([W, H]) / 2 # px and py

poses = poses[:,:3,:4]

print("loaded llff", images.shape, render_poses.shape, hwf, "./data/llff/fern")

########################## EXTRINSICS ##################################

# Extract camera extrinsic matrix
poses_file = './data/llff/fern/poses_bounds.npy'
poses = np.load(poses_file)

# camera extrinsics for each of the 20 images
extrinsics = poses[:, :12].reshape(-1, 3, 4)
# print(extrinsics)

###################### FOV, ASPECT RATIO ###############################

aspect_ratio = W / H
horiz_fov = 2 * np.arctan(W / (2 * focal))
fov = 2 * np.arctan(np.tan(horiz_fov / 2) * aspect_ratio)

######################## INTRINSIC MATRIX ##############################

# Compute camera intrinsic matrix (not sure if this is right tbh)
focal_length = hwf[2] / (2.0 * np.tan(0.5 * fov * np.pi / 180.0))
intrinsic_mat = np.array([[focal_length, 0.0, 0.5 * hwf[1]],
                          [0.0, focal_length * aspect_ratio, 0.5 * hwf[0]],
                          [0.0, 0.0, 1.0]])
print("intrinsic matrix\n", intrinsic_mat)
