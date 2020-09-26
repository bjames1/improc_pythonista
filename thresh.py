

"""
Author: James Michael Brown
Date: 09/24/2020
Note: Threshold image
"""

# # # Import Modules
from PIL import Image, ImageOps, ImageFilter
from PIL.Image import BILINEAR
from math import sqrt, sin, cos, atan2
import dialogs
import photos
import numpy as np # fundamental package for scientific computing
import matplotlib.pyplot as plt # package for plot function

all_assets = photos.get_assets()
last_asset = all_assets[-1]
img = last_asset.get_image()
img = ImageOps.grayscale(img)

album = photos.get_recently_added_album()
last = album.assets[-1]
if last.can_edit_content:
    img = last.get_image()
    grayscale_img = img.convert('L')
    grayscale_img.save('.edit.jpg', quality=90)
    last.edit_content('.edit.jpg')
else:
    print('The asset is not editable')

y=grayscale_img
plt.imshow(y)
plt.show()
plt.close('all')

th = 128
im_bool = y > th
#print(im_bool)
im_bin_128 = (y > th) * 255
im_bin_64 = (y > 64) * 255
im_bin_192 = (y > 192) * 255

plt.imshow(im_bin_128)
plt.show()
plt.close('all')

im_bin = np.concatenate((im_bin_64, im_bin_128, im_bin_192), axis=1)
Image.fromarray(np.uint8(im_bin)).save('binarization.png')
