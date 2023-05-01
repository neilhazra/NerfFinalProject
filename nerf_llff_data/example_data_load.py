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

poses = poses[:,:3,:4]

print("loaded llff", images.shape, render_poses.shape, hwf, "./data/llff/fern")

# Compute camera intrinsic matrix
focal_length = hwf[2] / \
    (2.0 * np.tan(0.5 * self.hparams['fov'] * np.pi / 180.0))
intrinsic_mat = np.array([[focal_length, 0.0, 0.5 * hwf[1]],
                          [0.0, focal_length *
                              self.hparams['aspect_ratio'], 0.5 * hwf[0]],
                          [0.0, 0.0, 1.0]])
print(intrinsic_mat)

########################## EXTRINSICS ##################################

# Extract camera extrinsic matrix
poses_file = './data/llff/fern/poses_bounds.npy'
poses = np.load(poses_file)

# camera extrinsics for each of the 20 images
extrinsics = poses[:, :12].reshape(-1, 3, 4)
# print(extrinsics)
