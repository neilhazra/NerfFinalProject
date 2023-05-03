import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from geometry_utils import *

# Load images
rgb = cv2.cvtColor(cv2.imread('../../../drums/test/r_0.png'), cv2.COLOR_BGR2RGB)
# rgb_other = cv2.cvtColor(cv2.imread('../../../drums/test/r_56.png'), cv2.COLOR_BGR2RGB)

# Depth is stored as float32 in meters
# WE ARE USING PNG FOR DEPTH - will cross the bridge when we get theres
depth = cv2.imread('../../../drums/test/r_0_depth_0001.png', cv2.IMREAD_ANYDEPTH)
# depth_other = cv2.imread('../../../drums/test/r_56_depth_0001.png', cv2.IMREAD_ANYDEPTH)

# transform matrix (extrinsics)
# transform_mat = np.array([[-1.0000001192092896,0.0,0.0,0.0],
#                 [0.0,-0.7341100573539734,0.67903071641922,2.737260103225708],
#                 [0.0,0.67903071641922,0.7341099977493286,2.959291696548462],
#                 [0.0,0.0,0.0,1.0]])
# transform_mat_other = np.array([[0.9297761917114258, -0.14997360110282898, 0.3361911177635193, 1.355229377746582], 
#                                 [0.36812570691108704, 0.3787887692451477, -0.8491188287734985, -3.4229068756103516],
#                                 [0.0, 0.913250744342804, 0.40739792585372925, 1.642273187637329],
#                                 [0.0, 0.0, 0.0, 1.0]])
# transform_inv = np.linalg.inv(transform_mat)
# transform_inv_other = np.linalg.inv(transform_mat_other)

# Get intrinsic parameters --> we got these from the intrinsics estimated in example_data_load.py
# height, width, _ = rgb.shape
# K = intrinsic_from_fov(height, width, 90)  # +- 45 degrees
K = np.array([[1.25000005e+03, 0.00000000e+00, 4.00000000e+02,0],
 [0.00000000e+00, 1.25000005e+03, 4.00000000e+02,0],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00,0],[0,0,0,1]])

K_inv = np.linalg.inv(K)
width = 800
height = 800

# Get pixel coordinates
pixel_coords = pixel_coord_np(width, height)  # [3, npoints]


# Apply back-projection: K_inv @ pixels * depth
cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()
# cam_coords_other = K_inv[:3, :3] @ pixel_coords * depth_other.flatten()

# transform matrices??

# transformed = transform_inv @ cam_coords
# transformed_other = transform_inv_other @ cam_coords_other


# back-projection using native for-loop.
# Uncomment block to test this
# cam_coords = np.zeros((height * width, 3))
# u0 = K[0, 2]
# v0 = K[1, 2]
# fx = K[0, 0]
# fy = K[1, 1]
# i = 0
# # Loop through each pixel in the image
# for v in range(height):
#     for u in range(width):
#         # Apply equation in fig 3
#         x = (u - u0) * depth[v, u] / fx
#         y = (v - v0) * depth[v, u] / fy
#         z = depth[v, u]
#         cam_coords[i] = (x, y, z)
#         i += 1
# cam_coords = cam_coords.T


# Limit points to 150m in the z-direction for visualisation
cam_coords = cam_coords[:, np.where(cam_coords[2] <= 150)[0]]
# cam_coords_other = cam_coords_other[:, np.where(cam_coords_other[2] <= 150)[0]]

# print(cam_coords.shape)
# print(cam_coords_other.shape)

# cam_coords_all = np.append(cam_coords, cam_coords_other,axis=1)

# print(cam_coords_all.shape)

# Visualize
pcd_cam = o3d.geometry.PointCloud()
pcd_cam.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
# Flip it, otherwise the pointcloud will be upside down
pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
o3d.visualization.draw_geometries([pcd_cam])

# # Visualize both
# pcd_cam = o3d.geometry.PointCloud()
# # pcd_cam.points = o3d.utility.Vector3dVector(cam_coords_all.T[:, :3])
# # Flip it, otherwise the pointcloud will be upside down
# pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# o3d.visualization.draw_geometries([pcd_cam])

# def project_topview(cam_points):
#     """
#     Draw the topview projection
#     """
#     max_longitudinal = 70
#     window_x = (-50, 50)
#     window_y = (-3, max_longitudinal)

#     x, y, z = cam_points
#     # flip the y-axis to positive upwards
#     y = - y

#     # We sample points for points less than 70m ahead and above ground
#     # Camera is mounted 1m above on an ego vehicle
#     ind = np.where((z < max_longitudinal) & (y > -1.2))
#     bird_eye = cam_points[:3, ind]

#     # Color by radial distance
#     dists = np.sqrt(np.sum(bird_eye[0:2:2, :] ** 2, axis=0))
#     axes_limit = 10
#     colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

#     # Draw Points
#     fig, axes = plt.subplots(figsize=(12, 12))
#     axes.scatter(bird_eye[0, :], bird_eye[2, :], c=colors, s=0.1)
#     axes.set_xlim(window_x)
#     axes.set_ylim(window_y)
#     axes.set_title('Bird Eye View')
#     plt.axis('off')

#     plt.gca().set_aspect('equal')
#     plt.show()


# # Do top view projection
# project_topview(cam_coords)
