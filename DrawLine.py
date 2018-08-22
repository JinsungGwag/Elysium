# import vision manage library
import cv2
# import data manage library
import pandas as pd

# csv data path
data_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\depth_img\export.csv'

# color image path
color_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\depth_color_img'

# save path
save_path = 'C:\Users\dkdjs\Desktop\Elysium\Resource\images\New\New'

# read csv data
dataframe = pd.read_csv(data_path, header=None, index_col=None)

# images name
img_name = ['9', '8', '20', '28', '34', '44', '52', '56', '62', '68', '70', '81', '79', '85', '89', '96', '4',
            '13', '39', '48', '73', '91', '94', '64', '42', '22', '25', '58', '49', '36']

for i in range(0, len(img_name)):

    cervical = (dataframe[1][i], dataframe[2][i])
    lumbar = (dataframe[3][i], dataframe[4][i])

    # load color image
    img = cv2.imread(color_path + '/' + img_name[i] + '.png')

    # draw line in image
    cv2.line(img, cervical, lumbar, (0, 0, 0), 3)

    # save image
    cv2.imwrite(save_path + '/' + img_name[i] + '_line.png', img)
