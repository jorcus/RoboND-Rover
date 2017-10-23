import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


def obstacle_thresh(img):
    threshed = color_thresh(img)
    return np.absolute(np.float32(threshed) - 1)

# function to identify the rocks
def rocks_thresh(img, rgb_thresh=(100,100,60)):
    rockpix = ((img[:,:,0] > rgb_thresh[0]) \
                  & (img[:,:,1] > rgb_thresh[1]) \
                  & (img[:,:,2] < rgb_thresh[2]))
    
    rock_select = np.zeros_like(img[:,:,0])
    rock_select[rockpix] = 1
    return rock_select


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])

    img = Rover.img
    dst_size = 5 
    bottom_offset = 6
    # 1) Define source and destination points for perspective transform
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    
    
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    color_threshed = color_thresh(warped)
    obstacle_threshed = obstacle_thresh(warped)
    rocks_threshed = rocks_thresh(warped)

    Rover.vision_image[:,:,2] = color_threshed * 255
    Rover.vision_image[:,:,0] = obstacle_threshed * 255
    
    # 4) Convert thresholded image pixel values to rover-centric coords
    nav_xpix, nav_ypix = rover_coords(color_threshed)
    obs_xpix, obs_ypix = rover_coords(obstacle_threshed)
    
    # 5) Convert rover-centric pixel values to world coords
    dist, angles = to_polar_coords(nav_xpix, nav_ypix)
    mean_dir = np.mean(angles)
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw
    world_shape = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    
    # 6) Update worldmap (to be displayed on right side of screen)
        # Example: data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          data.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          data.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    nav_x_world, nav_y_world = pix_to_world(nav_xpix, nav_ypix, xpos, ypos, yaw, world_shape, scale)
    obs_x, obs_y = pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_shape, scale)

    Rover.worldmap[nav_y_world, nav_x_world, 2] += 10
    Rover.worldmap[obs_y, obs_x, 0] += 1
    Rover.nav_angles = angles


    if rocks_threshed.any():
        rock_xpix, rock_ypix = rover_coords(rocks_threshed)
        rock_x, rock_y = pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_shape, scale)
        rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
        rock_idx = np.argmin(rock_dist)
        rock_xcen = rock_x
        rock_ycen = rock_y

        Rover.worldmap[rock_ycen, rock_xcen, :] = 255
        Rover.vision_image[:, :, 1] = rocks_threshed * 255
    else:
        Rover.vision_image[:, :, 1] = 0

    return Rover

