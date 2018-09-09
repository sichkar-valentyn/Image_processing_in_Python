# Image processing in Python
Experimental results at image processing in Python.
<br/>[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1343603.svg)](https://doi.org/10.5281/zenodo.1343603)

### Reference to:
Valentyn N Sichkar. Image processing in Python // GitHub platform. DOI: 10.5281/zenodo.1343603

### Related works:
* Sichkar V.N. Comparison analysis of knowledge based systems for navigation of mobile robot and collision avoidance with obstacles in unknown environment. St. Petersburg State Polytechnical University Journal. Computer Science. Telecommunications and Control Systems, 2018, Vol. 11, No. 2, Pp. 64–73. DOI: <a href="https://doi.org/10.18721/JCSTCS.11206" target="_blank">10.18721/JCSTCS.11206</a>

* The study of Neural Networks for Computer Vision in autonomous vehicles and robotics is put in separate repository and is available here: https://github.com/sichkar-valentyn/Neural_Networks_for_Computer_Vision

* The research on Machine Learning algorithms and techniques in Python is put in separate repository and is available here: https://github.com/sichkar-valentyn/Machine_Learning_in_Python

## Description
Image processing. Getting data from images in form of matrix with numbers, slicing them into color channels, applying filtering. Code examples with a lot of comments.

## Content
Codes (it'll send you to appropriate file):
* [Opening_png_jpg](https://github.com/sichkar-valentyn/Image_processing_in_Python/tree/master/Codes/Opening_png_jpg.py)
* [Converting_RGB_to_GreyScale](https://github.com/sichkar-valentyn/Image_processing_in_Python/tree/master/Codes/Converting_RGB_to_GreyScale.py)
* [Simple_Filtering](https://github.com/sichkar-valentyn/Image_processing_in_Python/tree/master/Codes/Simple_Filtering.py)

<br/>
Experimental results (figures and tables on this page):

* <a href="#RGB channels of the image separately">RGB channels of the image separately</a>
* <a href="#Examples of Simple Filtering">Examples of Simple Filtering</a>

<br/>

### <a name="RGB channels of the image separately">RGB channels of the image separately</a>

![RGB_channels](images/RGB_channels_of_image.png)

<br/>

### <a name="Examples of Simple Filtering">Examples of Simple Filtering</a>
Taking RGB image as input, converting it to GrayScale.
<br/>Consider following part of the code:

```py
# Creating an array from image data
input_image = Image.open("images/eagle.jpeg")
image_np = np.array(input_image)

# Preparing image for Edge detection
# Converting RGB image into GrayScale image
# Using formula:
# Y' = 0.299 R + 0.587 G + 0.114 B
image_GrayScale = image_np[:, :, 0] * 0.299 + image_np[:, :, 1] * 0.587 + image_np[:, :, 2] * 0.114
```

Setting Hyperparameters and applying Pad frame for input image.
<br/>Consider following part of the code:

```py
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
```

Declaring filters for **Edge Detection**.
<br/>Consider following part of the code:

```py
# Declaring standard filters (kernels) with size 3x3 for edge detection
filter_1 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]])
filter_2 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
filter_3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
```

Creating function to delete negative values from resulted image.
<br/>Consider following part of the code:

```py
# In order to prevent appearing values that are less than -1
# Following function is declared
def relu(array):
    # Preparing array for output result
    r = np.zeros_like(array)
    # Using 'np.where' setting condition that every element in 'array' has to be more than appropriate element in 'r'
    result = np.where(array > r, array, r)
    # Returning resulted array
    return result
```

Creating function to delete values that are more than 255.
<br/>Consider following part of the code:

```py
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
```

Implementing filtering for **Edge Detection** also known as **Convolution Operation**.
<br/>Consider following part of the code:

```py
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
```

Showing resulted images on the figure.

![Simple_filtering_with_convolution](images/Simple_filtering_with_convolution.png)

Full code is available here: [Simple_Filtering.py](https://github.com/sichkar-valentyn/Image_processing_in_Python/tree/master/Codes/Simple_Filtering.py)

<br/>

### MIT License
### Copyright (c) 2018 Valentyn N Sichkar
### github.com/sichkar-valentyn
### Reference to:
Valentyn N Sichkar. Image processing in Python // GitHub platform. DOI: 10.5281/zenodo.1343603
