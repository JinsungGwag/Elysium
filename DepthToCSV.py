# Import data manage library
import pandas as pd
# Import vision manage library
import cv2

# data save path
data_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\csv'

# depth image path
image_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\depth_img'

# depth image file name
image_name = '1'

# load depth image
img = cv2.imread(image_path + '/' + image_name + '.png', cv2.IMREAD_ANYDEPTH)

# save data in csv file
dataframe = pd.DataFrame(img)
dataframe.to_csv(data_path + '/data.csv', header=None, index=None)