from models.gdal_convert import ENVI_RAW
from models.cv_circles_linesP import recongize_circle,recongize_linesP
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

import tkinter as tk

root = tk.Tk()

pil_image = Image.fromarray(rgb)
w, h = pil_image.size
fname = 'None'
# split off image file name
sf = "{} ({}x{})".format(fname, w, h)
root.title(sf)

# convert PIL image object to Tkinter PhotoImage object
tk_image = ImageTk.PhotoImage(pil_image)

# put the image on a typical widget
label = tk.Label(root, image=tk_image, bg='brown')
label.pack(padx=5, pady=5)

root.mainloop()


# file = 'D:\\Desktop\\data\\BSQ\\306.raw.BSQ'
# file = 'Z:\\data\\BSQ\\306_ref.raw.BSQ'
raw = ENVI_RAW(file)
circle = recongize_circle(raw.get_gray())
print(circle.type)
circle.train_search_output()
rgb = circle.draw_all(raw.get_rgb())