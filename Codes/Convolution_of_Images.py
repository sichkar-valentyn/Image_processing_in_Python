# File: Convolution_of_Images.py
# Description: Image processing via convolution
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Image processing in Python // GitHub platform. DOI: 10.5281/zenodo.1343603




# Image processing via convolution
# Importing needed library
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

# Reading images
cat, dog = imread('images/cat.jpg'), imread('images/dog.jpg')

# Defining difference between width and height
print(cat.shape)  # (1080, 1920, 3)
print(dog.shape)  # (1050, 1680, 3)
difference_cat = cat.shape[1] - cat.shape[0]
difference_dog = dog.shape[1] - dog.shape[0]
# Cropping images to make it square size
# Cropping by width and taking middle part
cat_cropped = cat[:, int(difference_cat / 2):int(-difference_cat / 2), :]
dog_cropped = dog[:, int(difference_dog / 2):int(-difference_dog / 2), :]
print(cat_cropped.shape)  # (1080, 1080, 3)
print(dog_cropped.shape)  # (1050, 1050, 3)

# Defining needed image size for resizing
image_size = 200
# Defining output array for new images
# For 2 images with height = width = image_size and 3 channels
# (channels come at the end in order to show resized image)
image_resized = np.zeros((2, image_size, image_size, 3))
print(image_resized.shape)  # (2, 200, 200, 3)
# Resizing two images
image_resized[0, :, :, :] = imresize(cat_cropped, (image_size, image_size))  # (200, 200, 3)
image_resized[1, :, :, :] = imresize(dog_cropped, (image_size, image_size))  # (200, 200, 3)

# Preparing data for convolution operation
# Defining output array for new image
# For 2 images with 3 channels and height = width = image_size
x = np.zeros((2, 3, image_size, image_size))
# Resizing two images
# And transposing in order to put channels first
x[0, :, :, :] = imresize(cat_cropped, (image_size, image_size)).transpose((2, 0, 1))
x[1, :, :, :] = imresize(dog_cropped, (image_size, image_size)).transpose((2, 0, 1))
print(x[0].shape)  # (3, 200, 200)
print(x[1].shape)  # (3, 200, 200)

# Preparing weights for convolution for 2 filters with 3 channels and size 3x3
# Defining array for weights
w = np.zeros((2, 3, 3, 3))

# First filter converts images into grayscale
# Defining three channels for this filter - red, green and blue
w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

# Second filter will detect horizontal edges in the blue channel
w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# Defining 128 biases for the edge detection filter
# in order to make output non-negative
b = np.array([0, 128])


"""
Defining function for naive forward pass for convolutional layer
Input consists of following:
x of shape (N, C, H, W) - N data, each with C channels, height H and width W.
w of shape (F, C, HH, WW) - We convolve each input with F different filters,
where each filter spans all C channels; each filter has height HH and width WW.

'cnn_params' is a dictionary with following keys:
'stride' - step for sliding
'pad' - zero-pad frame around input

Function returns volume of feature maps of shape (N, F, H', W') where:
H' = 1 + (H + 2 * pad - HH) / stride
W' = 1 + (W + 2 * pad - WW) / stride

N here is the same as we have it as number of input images.
F here is as number of channels of each N (that are now as feature maps)

"""


def cnn_forward_naive(x, w, b, cnn_params):
    # Preparing parameters for convolution operation
    stride = cnn_params['stride']
    pad = cnn_params['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    # Applying to the input image volume Pad frame with zero values for all channels
    # As we have in input x N as number of inputs, C as number of channels,
    # then we don't have to pad them
    # That's why we leave first two tuples with 0 - (0, 0), (0, 0)
    # And two last tuples with pad parameter - (pad, pad), (pad, pad)
    # In this way we pad only H and W of N inputs with C channels
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Defining spatial size of output image volume (feature maps) by following formulas:
    height_out = int(1 + (H + 2 * pad - HH) / stride)
    width_out = int(1 + (W + 2 * pad - WW) / stride)
    # Depth of output volume is number of filters which is F
    # And number of input images N remains the same - it is number of output image volumes now

    # Creating zero valued volume for output feature maps
    feature_maps = np.zeros((N, F, height_out, width_out))

    # Implementing convolution through N input images, each with F filters
    # Also, with respect to C channels
    # For every image
    for n in range(N):
        # For every filter
        for f in range(F):
            # Defining variable for indexing height in output feature map
            # (because our step might not be equal to 1)
            height_index = 0
            # Convolving every channel of the image with every channel of the current filter
            # Result is summed up
            # Going through all input image (2D convolution) through all channels
            for i in range(0, H, stride):
                # Defining variable for indexing width in output feature map
                # (because our step might not be equal to 1)
                width_index = 0
                for j in range(0, W, stride):
                    feature_maps[n, f, height_index, width_index] = \
                        np.sum(x_padded[n, :, i:i+HH, j:j+WW] * w[f, :, :, :]) + b[f]
                    # Increasing index for width
                    width_index += 1
                # Increasing index for height
                height_index += 1

    # Returning resulted volumes of feature maps and cash
    return feature_maps


# Implementing convolution of each image with each filter and offsetting by bias
results = cnn_forward_naive(x, w, b, {'stride': 1, 'pad': 1})
print(results.shape)  # (2, 2, 200, 200) - two images with two channels


# Creating function for normalizing resulted images
def normalize_image(img):
    image_max, image_min = np.max(img), np.min(img)
    return 255 * (img - image_min) / (image_max - image_min)


# Preparing figures for plotting
figure_1, ax = plt.subplots(nrows=2, ncols=5)
# 'ax 'is as (2, 5) np array and we can call each time ax[0, 0]

# Plotting original, cropped and resized images
# By adding 'astype' we convert float numbers to integer
ax[0, 0].imshow(cat)
ax[0, 0].set_title('Original (900, 1600, 3))')
ax[0, 1].imshow(cat_cropped)
ax[0, 1].set_title('Cropped (900, 900, 3)')
ax[0, 2].imshow(image_resized[0, :, :, :].astype('int'))
ax[0, 2].set_title('Resized (200, 200, 3)')
ax[0, 3].imshow(normalize_image(results[0, 0]), cmap=plt.get_cmap('gray'))
ax[0, 3].set_title('Grayscale')
ax[0, 4].imshow(normalize_image(results[0, 1]), cmap=plt.get_cmap('gray'))
ax[0, 4].set_title('Edges')

ax[1, 0].imshow(dog)
ax[1, 0].set_title('Original (1050, 1680, 3)')
ax[1, 1].imshow(dog_cropped)
ax[1, 1].set_title('Cropped (1050, 1050, 3)')
ax[1, 2].imshow(image_resized[1, :, :, :].astype('int'))
ax[1, 2].set_title('Resized (200, 200, 3)')
ax[1, 3].imshow(normalize_image(results[1, 0]), cmap=plt.get_cmap('gray'))
ax[1, 3].set_title('Grayscale')
ax[1, 4].imshow(normalize_image(results[1, 1]), cmap=plt.get_cmap('gray'))
ax[1, 4].set_title('Edges')

# Setting axes 'off'
for i in range(2):
    for j in range(5):
        ax[i, j].set_axis_off()

# Giving the name to the window with figure
figure_1.canvas.set_window_title('Image convolution')
# Showing the plots
plt.show()
