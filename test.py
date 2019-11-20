from private_model.gdal_convert import ENVI_RAW
from private_model.cv_circles_linesP import recongize_circle,recongize_linesP
from matplotlib import pyplot as plt
# file = 'D:\\Desktop\\data\\BSQ\\306.raw.BSQ'
file = 'Z:\\data\\BSQ\\306_ref.raw.BSQ'

raw = ENVI_RAW(file)
circle = recongize_circle(raw.get_gray())
print(circle.type)
circle.train_search_output()
rgb = circle.draw_all(raw.get_rgb())

# plt initialize
plt.figure('hyperspectral image', figsize=(19.2, 10.8), dpi=100)
plt.suptitle('Hyperspectral Image', fontsize=40)
plt.imshow(rgb)
plt.title('rgb', verticalalignment='baseline',
          horizontalalignment='center', fontsize=10)
plt.xticks([]), plt.yticks([])
plt.show()