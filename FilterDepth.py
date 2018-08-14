# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import Pandas for save data
import pandas as pd
# Import Keyboard to detect key press
import keyboard as kb

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print "Depth Scale is: ", depth_scale

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
clipping_distance_in_meters = 2  # meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Both sides
leftBorder = 120
rightBorder = 200

# Image size
imgWidth = 320
imgHeight = 240

# Image division
imgDivision = 5

#  Post process frame
def post_process_depth_frame(depth_frame, decimation_magnitude=1.0, spatial_magnitude=2.0, spatial_smooth_alpha=0.5,
                             spatial_smooth_delta=20, temporal_smooth_alpha=0.4, temporal_smooth_delta=20):
    """
    Filter the depth frame acquired using the Intel RealSense device
    Parameters:
    -----------
    depth_frame 	 	 	 : rs.frame()
                               The depth frame to be post-processed
    decimation_magnitude : double
                              The magnitude of the decimation filter
    spatial_magnitude 	 : double
                            The magnitude of the spatial filter
    spatial_smooth_alpha	 : double
                            The alpha value for spatial filter based smoothening
    spatial_smooth_delta	 : double
                            The delta value for spatial filter based smoothening
    temporal_smooth_alpha : double
                            The alpha value for temporal filter based smoothening
    temporal_smooth_delta : double
                            The delta value for temporal filter based smoothening
    Return:
    ----------
    filtered_frame : rs.frame()
                       The post-processed depth frame
    """

    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    # Available filters and control options for the filters
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    filter_magnitude = rs.option.filter_magnitude
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta

    # Apply the control parameters for the filter
    decimation_filter.set_option(filter_magnitude, decimation_magnitude)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)

    return filtered_frame


# Streaming loop
try:
    while True:
        # Get frame set of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        post_process_depth_frame(aligned_depth_frame)

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Resize images
        resize_depth_image = cv2.resize(depth_image, (imgWidth, imgHeight), fx=0.5, fy=0.5,
                                        interpolation=cv2.INTER_AREA)
        resize_color_image = cv2.resize(color_image, (imgWidth, imgHeight), fx=0.5, fy=0.5,
                                        interpolation=cv2.INTER_AREA)

        # Save csv data
        if kb.is_pressed('s'):
            data = pd.DataFrame(resize_depth_image[:, imgHeight / 2 - 20:imgHeight / 2 + 20])
            data.to_csv("C:\Users\dkdjs\Desktop\Elysium\Data\output.csv", mode='w')

        # depth_image is height x width array

        # Save max depth points
        imgUnitHeight = imgHeight / imgDivision
        arr = np.zeros(shape=(imgUnitHeight, 2))
        for i in range(0, imgUnitHeight):
            arr[i] = [leftBorder + np.argmax(resize_depth_image[i * imgDivision][leftBorder:rightBorder]), i *
                      imgDivision]

        # Convert to numpy arrays
        arr = np.array(arr, np.int32)
        arr = arr.reshape((-1, 1, 2))

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (resize_depth_image, resize_depth_image, resize_depth_image))
        # depth image is 1 channel, color is 3 channels

        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color,
                              resize_color_image)

        # Draw max depth line
        # cv2.polylines(bg_removed, [arr], False, (180, 0, 0), 2)

        # Draw max depth point
        for i in range(0, imgUnitHeight):
            if arr[i][0][0] != leftBorder:
                cv2.circle(bg_removed, (arr[i][0][0], arr[i][0][1]), 3, (180, 0, 0), 1)

        # Draw range line
        # cv2.line(bg_removed, (280, 0), (280, 479), (0, 0, 180), 3)
        # cv2.line(bg_removed, (360, 0), (360, 479), (0, 0, 180), 3)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(resize_depth_image, alpha=0.5), cv2.COLORMAP_JET)

        # Draw range line
        cv2.line(depth_colormap, (leftBorder, 0), (leftBorder, imgHeight - 1), (0, 0, 0), 3)
        cv2.line(depth_colormap, (rightBorder, 0), (rightBorder, imgHeight - 1), (0, 0, 0), 3)

        # Show depth and color images
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        cv2.waitKey(1)

finally:
    pipeline.stop()
