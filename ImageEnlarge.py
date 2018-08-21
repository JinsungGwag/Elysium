# Import image utility
import imutils as iu
# Import opencv
import cv2
# Import pandas
import pandas as pd
# Import math
import math
# Import numpy
import numpy as np

# depth image path
image_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\depth_img'

# color image path
color_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\color_img'

# csv data path
data_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\depth_img\data.csv'

# save path
save_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\New'

# image file size
file_size = 99

# load data from directory
dataframe = pd.read_csv(data_path, header=None, index_col=None)

# Convert dataframe to numpy array
array = np.array(dataframe)

# rotate data array
new_data = []


# Rotate point about point
def rotate(origin, point, angle):

    angle = math.radians(angle)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    qx = int(qx)
    qy = int(qy)

    return qx, qy


# enlarge images
for i in range(0, file_size):

    img = cv2.imread(image_path + '/{:0}.png'.format(i+1), cv2.IMREAD_ANYDEPTH)
    col_img = cv2.imread(color_path + '/{:0}.png'.format(i+1))

    # rotate image
    rotate_img1 = iu.rotate(img, -5)
    rotate_img2 = iu.rotate(img, 5)

    # rotate color image
    rot_col_img1 = iu.rotate(col_img, -5)
    rot_col_img2 = iu.rotate(col_img, 5)

    # rotate point
    cervical_point_1 = (rotate((160, 120), (dataframe[1][i], dataframe[2][i]), 5))
    lumbar_point_1 = (rotate((160, 120), (dataframe[4][i], dataframe[5][i]), 5))
    cervical_point_2 = (rotate((160, 120), (dataframe[1][i], dataframe[2][i]), -5))
    lumbar_point_2 = (rotate((160, 120), (dataframe[4][i], dataframe[5][i]), -5))

    # draw lines at color image
    cv2.line(rot_col_img1, cervical_point_1, lumbar_point_1, (255, 255, 255), 3)
    cv2.line(rot_col_img2, cervical_point_2, lumbar_point_2, (255, 255, 255), 3)

    # save images
    cv2.imwrite(save_path + '/' + str(i + 1) + '_1.png', rotate_img1)
    cv2.imwrite(save_path + '/' + str(i + 1) + '_2.png', rotate_img2)

    # save color images
    cv2.imwrite(save_path + '/' + str(i + 1) + '_rotate1.png', rot_col_img1)
    cv2.imwrite(save_path + '/' + str(i + 1) + '_rotate2.png', rot_col_img2)

    # make new data array
    new_data.append(array[i, :])
    new_data.append(['C:/PycharmProjects/depth_img/' + str(i + 1) + '_1.png', cervical_point_1[0], cervical_point_1[1],
                     dataframe[3][i], lumbar_point_1[0], lumbar_point_1[1], dataframe[6][i]])
    new_data.append(['C:/PycharmProjects/depth_img/' + str(i + 1) + '_2.png', cervical_point_2[0], cervical_point_2[1],
                     dataframe[3][i], lumbar_point_2[0], lumbar_point_2[1], dataframe[6][i]])

# save dataframe to csv
new_dataframe = pd.DataFrame(new_data)
new_dataframe.to_csv(save_path + '/newData.csv', header=None, index=None)
