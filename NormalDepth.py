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
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print "Depth Scale is: ", depth_scale

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
clipping_distance_in_meters = 2  # 2 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Both sides
leftBorder = 240
rightBorder = 400

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

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Save csv data
        if kb.is_pressed('s'):
            data = pd.DataFrame(depth_image[:, 200:280])
            data.to_csv("C:\Users\dkdjs\Desktop\Elysium\Data\output.csv", mode='w')

        # depth_image is 480 x 640 image

        # Save max depth points
        arr = np.zeros(shape=(96, 2))
        for i in range(0, 96):
            arr[i] = [leftBorder + np.argmax(depth_image[i * 5][leftBorder:rightBorder]), i * 5]

        # Convert to numpy arrays
        arr = np.array(arr, np.int32)
        arr = arr.reshape((-1, 1, 2))

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels

        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Draw max depth line
        # cv2.polylines(bg_removed, [arr], False, (180, 0, 0), 2)

        # Draw max depth point
        for i in range(0, 95):
            if arr[i][0][0] != leftBorder:
                cv2.circle(bg_removed, (arr[i][0][0], arr[i][0][1]), 3, (180, 0, 0), 1)

        # Draw range line
        # cv2.line(bg_removed, (280, 0), (280, 479), (0, 0, 180), 3)
        # cv2.line(bg_removed, (360, 0), (360, 479), (0, 0, 180), 3)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

        # Draw range line
        cv2.line(depth_colormap, (leftBorder, 0), (leftBorder, 479), (0, 0, 0), 3)
        cv2.line(depth_colormap, (rightBorder, 0), (rightBorder, 479), (0, 0, 0), 3)

        # Show depth and color images
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        cv2.waitKey(1)

finally:
    pipeline.stop()