"""
Author: James Michael Brown
Date: 09/24/2020
Note: Gabor edge detection code adapted for Pythonista, Python 3.6.


"Pythonista is a complete development environment for writing Python™ 
scripts on your iPad or iPhone. Lots of examples are included — from games 
and animations to plotting, image manipulation, custom user interfaces, 
and automation scripts.

In addition to the powerful standard library, Pythonista provides extensive
support for interacting with native iOS features, like contacts, reminders,
photos, location data, and more."

http://omz-software.com/pythonista/

"""

# # # Import Modules
from PIL import Image, ImageOps, ImageFilter
from PIL.Image import BILINEAR
from math import sqrt, sin, cos, atan2
import dialogs
import photos
import numpy as np # fundamental package for scientific computing
import matplotlib.pyplot as plt # package for plot function

# # # Define Functions
def myimshow(I, **kwargs):
    # utility function to show image
    plt.figure();
    plt.axis('off')
    plt.imshow(I, cmap=plt.gray(), **kwargs)
    plt.show()

def genSinusoid(sz, A, omega, rho):
    # Generate Sinusoid grating
    # sz: size of generated image (width, height)
    radz = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radz[0], radz[0]+1), range(-radz[1], radz[1]+1)) # a BUG is fixed in this line

    stimuli = A * np.cos(omega[0] * x  + omega[1] * y + rho)
    return stimuli

theta = np.pi/4
omega = [np.cos(theta), np.sin(theta)]
sinusoidParam = {'A':1,'omega':omega,'rho':np.pi/2, 'sz':(32,32)}
myimshow(genSinusoid(**sinusoidParam))
# ** is a special syntax in python, which enables passing a key-value dictionary as paramet

myconst=.15
def genGabor(sz, omega, theta, func=np.cos,K=np.pi):
	radius = (int(sz[0]/2.0), int(sz[1]/2.0))
	[x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))
	x1 = x * np.cos(theta) + y * np.sin(theta)
	x2 = myconst
	x3= ((-x * np.sin(theta) + y * np.cos(theta))**2)
	x1 = (x1+x2*x3)
	y1 = -x * np.sin(theta) + y * np.cos(theta)
	gauss = omega**2 / (4*np.pi * K**2) * np.exp(-omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
	s1 = func(omega * x1)
	s2 = np.exp(K**2/2)
	sinusoid = s1 * s2
	gabor = gauss * sinusoid
	return gabor

# # # MAKE FILTER BANKS
theta = np.arange(0, np.pi, np.pi/4) # range of theta
omega = np.arange(0.2, 0.6, 0.1) # range of omega
params = [(t,o) for o in omega for t in theta]
sinFilterBank = []
cosFilterBank = []
gaborParams = []
for (theta, omega) in params:
    gaborParam = {'omega':omega, 'theta':theta, 'sz':(256, 256)}
    sinGabor = genGabor(func=np.sin, **gaborParam)
    cosGabor = genGabor(func=np.cos, **gaborParam)
    sinFilterBank.append(sinGabor)
    cosFilterBank.append(cosGabor)
    gaborParams.append(gaborParam)

plt.figure()
n = len(sinFilterBank)
for i in range(n):
	plt.subplot(4,4,i+1)
	#title(r'$\theta$={theta:.2f}$\omega$={omega}'.format(**gaborParams[i]))
	plt.axis('off'); plt.imshow(sinFilterBank[i])

plt.figure()
for i in range(n):
	plt.subplot(4,4,i+1)
#	title(r'$\theta$={theta:.2f}$\omega$={omega}'.format(**gaborParams[i]))
	plt.axis('off'); plt.imshow(cosFilterBank[i])
	plt.show()
plt.close()


age[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) + bias
    return new_image

all_assets = photos.get_assets()
last_asset = all_assets[-1]
img = last_asset.get_image()
img = ImageOps.grayscale(img)

import photos
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

for i in range(len(sinFilterBank)):
	sinGabor=cosFilterBank[i]
	y=np.asarray(y)
	z = np.fft.irfft2(np.fft.rfft2(sinGabor) * np.fft.rfft2(y, sinGabor.shape))
	plt.imshow(z, cmap="gray")
	plt.show()
