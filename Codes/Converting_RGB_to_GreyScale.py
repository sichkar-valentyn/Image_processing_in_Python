# File: Converting_RGB_to_GreyScale.py
# Description: Opening RGB image as array, converting to GreyScale and saving result into new file
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Image processing in Python // GitHub platform. DOI: 10.5281/zenodo.1343603




# Opening RGB image as array, converting to GreyScale and saving result into new file

# Importing needed libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import scipy.misc


# Creating an array from image data
image_RGB = Image.open("images/eagle.jpg")
image_np = np.array(image_RGB)

# Checking the type of the array
print(type(image_np))  # <class 'numpy.ndarray'>
# Checking the shape of the array
print(image_np.shape)

# Showing image with every channel separately
channel_R = image_np[:, :, 0]
channel_G = image_np[:, :, 1]
channel_B = image_np[:, :, 2]

# Creating a figure with subplots
f, ax = plt.subplots(nrows=2, ncols=2)
# ax is (2, 2) np array and to make it easier to read we use 'flatten' function
# Or we can call each time ax[0, 0]
ax0, ax1, ax2, ax3 = ax.flatten()

# Adjusting first subplot
ax0.imshow(channel_R, cmap='Reds')
ax0.set_xlabel('')
ax0.set_ylabel('')
ax0.set_title('Red channel')

# Adjusting second subplot
ax1.imshow(channel_G, cmap='Greens')
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_title('Green channel')

# Adjusting third subplot
ax2.imshow(channel_B, cmap='Blues')
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('Blue channel')

# Adjusting fourth subplot
ax3.imshow(image_np)
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_title('Original image')

# Function to make distance between figures
plt.tight_layout()
# Giving the name to the window with figure
f.canvas.set_window_title('Eagle image in three channels R, G and B')
# Showing the plots
plt.show()


# Converting RGB image into GrayScale image
# Using formula:
# Y' = 0.299 R + 0.587 G + 0.114 B
image_RGB = Image.open("images/eagle.jpg")
image_np = np.array(image_RGB)
image_GreyScale = image_np[:, :, 0] * 0.299 + image_np[:, :, 1] * 0.587 + image_np[:, :, 2] * 0.114
# Checking the type of the array
print(type(image_GreyScale))  # <class 'numpy.ndarray'>
# Checking the shape of the array
print(image_GreyScale.shape)
# Giving the name to the window with figure
plt.figure('GreyScaled image from RGB')
# Showing the image by using obtained array
plt.imshow(image_GreyScale, cmap='Greys')
plt.show()
# Preparing array for saving - creating three channels with the same data in each
# Firstly, creating array with zero elements
# And by 'image_GreyScale.shape + tuple([3])' we add one more element '3' to the tuple
# Now the shape will be (1080, 1920, 3) - which is tuple type
image_GreyScale_with_3_channels = np.zeros(image_GreyScale.shape + tuple([3]))
# Secondly, reshaping GreyScale image from 2D to 3D
x = image_GreyScale.reshape((1080, 1920, 1))
# Finally, writing all data in three channels
image_GreyScale_with_3_channels[:, :, 0] = x[:, :, 0]
image_GreyScale_with_3_channels[:, :, 1] = x[:, :, 0]
image_GreyScale_with_3_channels[:, :, 2] = x[:, :, 0]
# Saving image into a file from obtained 3D array
scipy.misc.imsave("images/result_1.jpg", image_GreyScale_with_3_channels)
# Checking that image was written with three channels and they are identical
result_1 = Image.open("images/result_1.jpg")
result_1_np = np.array(result_1)
print(result_1_np.shape)
print(np.array_equal(result_1_np[:, :, 0], result_1_np[:, :, 1]))
print(np.array_equal(result_1_np[:, :, 1], result_1_np[:, :, 2]))
# Showing saved resulted image
# Giving the name to the window with figure
plt.figure('GreyScaled image from RGB')
# Here we don't need to specify the map like cmap='Greys'
plt.imshow(result_1_np)
plt.show()


# Another way to convert RGB image into GreyScale image
image_RGB = io.imread("images/eagle.jpg")
image_GreyScale = color.rgb2gray(image_RGB)
# Checking the type of the array
print(type(image_GreyScale))  # <class 'numpy.ndarray'>
# Checking the shape of the array
print(image_GreyScale.shape)
# Giving the name to the window with figure
plt.figure('GreyScaled image from RGB')
# Showing the image by using obtained array
plt.imshow(image_GreyScale, cmap='Greys')
plt.show()
# Saving converted image into a file from processed array
scipy.misc.imsave("images/result_2.jpg", image_GreyScale)


# One more way for converting
image_RGB_as_GreyScale = io.imread("images/eagle.jpg", as_gray=True)
# Checking the type of the array
print(type(image_RGB_as_GreyScale))  # <class 'numpy.ndarray'>
# Checking the shape of the array
print(image_RGB_as_GreyScale.shape)
# Giving the name to the window with figure
plt.figure('GreyScaled image from RGB')
# Showing the image by using obtained array
plt.imshow(image_RGB_as_GreyScale, cmap='Greys')
plt.show()
# Saving converted image into a file from processed array
scipy.misc.imsave("images/result_3.jpg", image_RGB_as_GreyScale)
