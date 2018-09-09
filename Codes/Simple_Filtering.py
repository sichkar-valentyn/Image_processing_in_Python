# File: Simple_Filtering.py
# Description: Example on simple filtering for edge detection
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Image processing in Python // GitHub platform. DOI: 10.5281/zenodo.1343603




# Input RGB image and implementing simple filtering

# Importing needed libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Creating an array from image data
input_image = Image.open("images/eagle.jpeg")
image_np = np.array(input_image)

# Checking the type of the array
print(type(image_np))  # <class 'numpy.ndarray'>
# Checking the shape of the array
print(image_np.shape)  # (270, 480, 3)

# Showing image with every channel separately
channel_0 = image_np[:, :, 0]
channel_1 = image_np[:, :, 1]
channel_2 = image_np[:, :, 2]

# Checking if all channels are different
print(np.array_equal(channel_0, channel_1))  # False
print(np.array_equal(channel_1, channel_2))  # False

# Creating a figure with subplots
f, ax = plt.subplots(nrows=2, ncols=2)
# ax is (2, 2) np array and to make it easier to read we use 'flatten' function
# Or we can call each time ax[0, 0]
ax0, ax1, ax2, ax3 = ax.flatten()

# Adjusting first subplot
ax0.imshow(channel_0, cmap=plt.get_cmap('Reds'))
ax0.set_xlabel('')
ax0.set_ylabel('')
ax0.set_title('First channel')

# Adjusting second subplot
ax1.imshow(channel_1, cmap=plt.get_cmap('Greens'))
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.set_title('Second channel')

# Adjusting third subplot
ax2.imshow(channel_2, cmap=plt.get_cmap('Blues'))
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.set_title('Third channel')

# Adjusting fourth subplot
ax3.imshow(image_np)
ax3.set_xlabel('')
ax3.set_ylabel('')
ax3.set_title('Original image')

# Function to make distance between figures
plt.tight_layout()
# Giving the name to the window with figure
f.canvas.set_window_title('GreyScaled image with three identical channels')
# Showing the plots
plt.show()

# Preparing image for Edge detection
# Converting RGB image into GrayScale image
# Using formula:
# Y' = 0.299 R + 0.587 G + 0.114 B
image_GrayScale = image_np[:, :, 0] * 0.299 + image_np[:, :, 1] * 0.587 + image_np[:, :, 2] * 0.114
# Checking the type of the array
print(type(image_GrayScale))  # <class 'numpy.ndarray'>
# Checking the shape of the array
print(image_GrayScale.shape)  # (270, 480)
# Giving the name to the window with figure
plt.figure('GrayScale image from RGB')
# Showing the image by using obtained array
plt.imshow(image_GrayScale, cmap=plt.get_cmap('gray'))
plt.show()

# Applying to the GrayScale image Pad frame with zero values
# Using NumPy method 'pad'
GrayScale_image_with_pad = np.pad(image_GrayScale, (1, 1), mode='constant', constant_values=0)
# Checking the shape
print(GrayScale_image_with_pad.shape)  # (272, 482)

# Preparing image for convolution
# In order to get filtered image (convolved input image) in the same size, it is needed to set Hyperparameters
# Filter (kernel) size, K_size = 3
# Step for sliding (stride), Step = 1
# Processing edges (zero valued frame around image), Pad = 1
# Consequently, output image size is (width and height are the same):
# Width_Out = (Width_In - K_size + 2*Pad)/Step + 1
# Imagine, that input image is 5x5 spatial size (width and height), then output image:
# Width_Out = (5 - 3 + 2*1)/1 + 1 = 5, and this is equal to input image

# Preparing zero valued output arrays for filtered images (convolved images)
# The shape is the same with input image according to the chosen Hyperparameters
# For three filters for Edge detection and implementing it only for one GrayScale channel
output_image_1 = np.zeros(image_GrayScale.shape)
output_image_2 = np.zeros(image_GrayScale.shape)
output_image_3 = np.zeros(image_GrayScale.shape)

# Declaring standard filters (kernel) with size 3x3 for edge detection
filter_1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
filter_2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
filter_3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# Checking the shape
print(filter_1.shape, filter_2.shape, filter_3.shape)
# ((3, 3) (3, 3) (3, 3)


# In order to prevent appearing values that are less than -1
# Following function is declared
def relu(array):
    # Preparing array for output result
    r = np.zeros_like(array)
    # Using 'np.where' setting condition that every element in 'array' has to be more than appropriate element in 'r'
    result = np.where(array > r, array, r)
    # Returning resulted array
    return result


# In order to prevent appearing values that are more than 255
# The following function is declared
def image_pixels(array):
    # Preparing array for output result
    # Creating an empty array
    r = np.empty(array.shape)
    # Filling array with 255 value for all elements
    r.fill(255)
    # Using 'np.where' setting condition that every element in 'array' has to be less than appropriate element in 'r'
    result = np.where(array < r, array, r)
    # Returning resulted array
    return result


# Implementing convolution operation for Edge detection for GrayScale image
# Going through all input image with pad frame
for i in range(GrayScale_image_with_pad.shape[0] - 2):
    for j in range(GrayScale_image_with_pad.shape[1] - 2):
        # Extracting 3x3 patch (the same size with filter) from input image with pad frame
        patch_from_input_image = GrayScale_image_with_pad[i:i+3, j:j+3]
        # Applying elementwise multiplication and summation - this is convolution operation
        # With filter_1
        output_image_1[i, j] = np.sum(patch_from_input_image * filter_1)
        # With filter_2
        output_image_2[i, j] = np.sum(patch_from_input_image * filter_2)
        # With filter_3
        output_image_3[i, j] = np.sum(patch_from_input_image * filter_3)

# Applying 'relu' and 'image_pixels' function to get rid of negative values and that ones that more than 255
output_image_1 = image_pixels(relu(output_image_1))
output_image_2 = image_pixels(relu(output_image_2))
output_image_3 = image_pixels(relu(output_image_3))

# Showing results on the appropriate figures
figure_1, ax = plt.subplots(nrows=3, ncols=1)

# Adjusting first subplot
ax[0].imshow(output_image_1, cmap=plt.get_cmap('gray'))
ax[0].set_xlabel('')
ax[0].set_ylabel('')
ax[0].set_title('Edge #1')

# Adjusting second subplot
ax[1].imshow(output_image_2, cmap=plt.get_cmap('gray'))
ax[1].set_xlabel('')
ax[1].set_ylabel('')
ax[1].set_title('Edge #2')

# Adjusting third subplot
ax[2].imshow(output_image_3, cmap=plt.get_cmap('gray'))
ax[2].set_xlabel('')
ax[2].set_ylabel('')
ax[2].set_title('Edge #3')

# Function to make distance between figures
plt.tight_layout()
# Giving the name to the window with figure
figure_1.canvas.set_window_title('Convolution with filters (simple filtering)')
# Showing the plots
plt.show()
