# Testing to open images and applying simple filtering

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
