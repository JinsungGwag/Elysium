# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import Keyboard to detect key press
import keyboard as kb\

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
clipping_distance_in_meters = 2  # 2 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Image size
imgWidth = 320
imgHeight = 240


# Send depth image to deep learning
def learn_image(depth_image):

    # 2 points in spine
    cervical = (0, 0)
    lumbar = (320, 240)

    return cervical, lumbar


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

        # Resize images
        resize_depth_image = cv2.resize(depth_image, (imgWidth, imgHeight), fx=0.5, fy=0.5,
                                        interpolation=cv2.INTER_AREA)
        resize_color_image = cv2.resize(color_image, (imgWidth, imgHeight), fx=0.5, fy=0.5,
                                        interpolation=cv2.INTER_AREA)

        # Use deep learning to detect spine
        if kb.is_pressed('s'):

            # Get spine point
            spine_points = learn_image(depth_image)

            # Change to gray scale
            gray_image = cv2.cvtColor(resize_color_image, cv2.COLOR_BGR2GRAY)

            # Draw line
            cv2.line(gray_image, spine_points[0], spine_points[1], (255, 255, 255), 3)

            # Show depth and color images
            cv2.namedWindow('Spine Detection', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Spine Detection', gray_image)

        # Depth_image is height x width array

        # Show depth and color images
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', color_image)
        cv2.waitKey(1)

finally:
    pipeline.stop()
