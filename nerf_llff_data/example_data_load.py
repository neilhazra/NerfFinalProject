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

############################ PIXEL SIZE ################################

# The default value for sensor_size is set to 36, which corresponds to a
#  full-frame camera sensor size of 36mm x 24mm. 
# Why 36? --> corresponds to the full-frame sensor size used in many 
# professional DSLR cameras
# focal length is also given in mm

sensor_size = 36
pixel_size = sensor_size / np.array([W, H]) * focal_length
# This gives the pixel size in millimeters per pixel.

# Explanation of the formula...

# The formula for pixel_size assumes that the camera sensor is rectangular 
# and that the image has the same aspect ratio as the sensor. This means 
# that the ratio of the sensor width to the sensor height is the same as 
# the ratio of the image width to the image height. With this assumption,
#  we can compute the physical size of a single pixel on the sensor by 
# dividing the sensor size by the number of pixels along each dimension, 
# and then scaling by the focal length to account for the projection of 
# the image onto the sensor.
