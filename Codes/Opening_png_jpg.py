# File: Opening_png_jpg.py
# Description: Testing to open images of different format
# Environment: PyCharm and Anaconda environment
#
# MIT License
# Copyright (c) 2018 Valentyn N Sichkar
# github.com/sichkar-valentyn
#
# Reference to:
# Valentyn N Sichkar. Image processing in Python // GitHub platform. DOI: 10.5281/zenodo.1343603




# Testing to open images of different format

import tkinter as tk
from PIL import Image, ImageTk

# Option 1 - opening .png images
# Establishing 'root' as the 'Tk' window
root = tk.Tk()
# Uploading image from file
image_1 = tk.PhotoImage(file="images/eagle.png")
# Creating a 'Label' object that contains uploaded image
label = tk.Label(root, image=image_1)
# Packing 'Label' into 'Tk' window
label.pack()
# Setting the title
root.title('Opening .png image')
# root.geometry("800x600")
# Starts loop with event
root.mainloop()


# Option 2 - opening .jpg images
# Establishing 'root' as the 'Tk' window
root = tk.Tk()
# Uploading image from file
image_2 = Image.open("images/eagle.jpg")
# Creating a 'Label' object that contains uploaded image
label = tk.Label(root)
label.image = ImageTk.PhotoImage(image_2)
label['image'] = label.image
# Packing 'Label' into 'Tk' window
label.pack()
# Setting the title
root.title('Opening .jpg image')
# Starts loop with event
root.mainloop()
