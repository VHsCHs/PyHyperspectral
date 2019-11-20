import cv2
import numpy as np
from skimage import exposure, img_as_float, img_as_ubyte

class recongize_circle():
    def __init__(self, garyimg):
        self.img = np.array(garyimg)
        self.circle = None
        self.type = self.img.dtype
    def train(self):
        # self.img = self.img / 256
        self.img = img_as_float(self.img)
        self.img = exposure.rescale_intensity(self.img)
        self.img = self.img * 256
        while exposure.is_low_contrast(self.img):
            self.img = exposure.adjust_gamma(self.img, 0.8)
        self.img = self.img.astype(np.uint8)
        self.img = img_as_ubyte(self.img)
        '''cv2 median filter-->>Reduce Noise'''
        '''Optional:cv2 gaussian filter-->>cv2.GaussianBlur'''
        self.img = cv2.medianBlur(self.img, 3)
        '''cv2 threshold-->>gain contrast'''
        '''Optional:cv2.adaptiveThreshold-->>have no good result'''
        ret, self.img = cv2.threshold(self.img, 50, 255, cv2.THRESH_BINARY)

    def search(self):
        '''
        Hough Circles recognize sample cell
        '''
        self.circles1 = cv2.HoughCircles(self.img, cv2.HOUGH_GRADIENT, 2, 100, param1=100, param2=25, minRadius=230,
                                         maxRadius=280)

    def output(self):
        '''
               Exception Handling
                   hough circles NOT identify circles
               Exception:
                   TypeError
               '''
        try:
            circles = self.circles1[0, :, :]
        except TypeError:
            pass
            print('no circles')
        else:
            circles = np.uint16(np.around(circles))
            temp = circles
            if np.shape(temp)[0] == 1:
                self.circle = temp[0]
            else:
                countarray = []
                for i in temp:
                    sumarray = []
                    x = i[0]
                    y = i[1]
                    r = i[2]
                    for yi, xline in enumerate(self.img):
                        for xi, value in enumerate(xline):
                            left = ((x - xi) ** 2 + (y - yi) ** 2)
                            right = r ** 2
                            if left <= right:
                                sumarray.append(value)
                    sumarray = np.array(sumarray)
                    sum = np.mean(sumarray)
                    countarray.append(sum)
                countarray = np.array(countarray)
                index = np.where(countarray == np.min(countarray))
                self.circle = temp[index[0][0]]
        return self.circle

    def train_search_output(self):
        self.train(), self.search(), self.output()

    def draw_all(self,color_img):
        if np.array(color_img).dtype == 'float32':
            color_img = color_img*256
        if np.array(color_img).dtype == 'uint16':
            color_img = color_img/256
        color_img = color_img.astype(np.uint8)
        for circle in self.circles1[0, :, :]:
            cv2.circle(color_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 5)
            cv2.circle(color_img, (circle[0], circle[1]), 150, (0, 0, 255), 5)
            cv2.circle(color_img, (circle[0], circle[1]), 6, (255, 0, 0), -1)
        return color_img

    def draw(self,color_img):
        if np.array(color_img).dtype == 'float32':
            color_img = color_img*256
        if np.array(color_img).dtype == 'uint16':
            color_img = color_img/256
        color_img = color_img.astype(np.uint8)
        cv2.circle(color_img, (self.circle[0], self.circle[1]), self.circle[2], (0, 255, 0), 5)
        cv2.circle(color_img, (self.circle[0], self.circle[1]), 150, (0, 0, 255), 5)
        cv2.circle(color_img, (self.circle[0], self.circle[1]), 6, (255, 0, 0), -1)
        return color_img

class recongize_linesP():
    def __init__(self, cannyimg):
        self.img = np.array(cannyimg)
        self.line = None
        self.type = self.img.dtype

    def train(self):
        # self.img = self.img / 256
        self.img = img_as_float(self.img)
        self.img = exposure.rescale_intensity(self.img)
        self.img = self.img * 256
        while exposure.is_low_contrast(self.img):
            self.img = exposure.adjust_gamma(self.img, 0.8)
        self.img = self.img.astype(np.uint8)
        # self.img = img_as_ubyte(self.img)
        '''cv2 median filter-->>Reduce Noise'''
        '''Optional:cv2 gaussian filter-->>cv2.GaussianBlur'''
        # self.img = cv2.medianBlur(self.img, 3)
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
        # self.img = cv2.fastNlMeansDenoising(self.img,h=3,templateWindowSize=7,searchWindowSize=21)
        '''cv2 threshold-->>gain contrast'''
        '''Optional:cv2.adaptiveThreshold-->>have no good result'''
        ret, self.img = cv2.threshold(self.img, 40, 255, cv2.THRESH_BINARY)
        '''cv2 canny-->>find high contrast area'''
        self.img = cv2.Canny(self.img, 90, 25)

    def search(self):
        '''cv2 HoughLinesP-->>find lines'''
        self.lines1 = cv2.HoughLinesP(self.img, 1, np.pi / 360, 50, minLineLength=100, maxLineGap=30)

    def Linear_X(self, y, b, k):
        x = (y - b) / k
        return int(x)

    def output(self, top_point=175, bot_point=675, L_offset=-135, R_offset=-35):
        rectangle_point = []
        try:
            len(self.lines1)
        except TypeError:
            print('no lines')
        else:
            for i in self.lines1:
                for x1, y1, x2, y2 in i:
                    if abs((y1 - y2) / (x1 - x2 + 1E-04)) > 7:
                        K = (y1 - y2) / (x1 - x2 + 1E-04)
                        if K < 0:
                            B = (x2 * y1 - y2 * x1) / (x2 - x1 + 1E-04)
                            X = self.Linear_X(0, B, K)

                            rectangle_point.append([B, K, x1, x2, y1, y2, X])
        rectangle_point = np.array(rectangle_point)
        try:
            rectangle_point_index = np.where(rectangle_point == np.min(rectangle_point[:, 6]))
        except IndexError:
            rectangle_point = [10000, -40]
        else:
            rectangle_point = rectangle_point[rectangle_point_index[0][0]]
        self._B = int(rectangle_point[0])
        self._K = int(rectangle_point[1])
        self.xmin = (self.Linear_X(bot_point, self._B, K)) + L_offset
        self.ymin = bot_point
        self.xmax = (self.Linear_X(top_point, self._B, K)) + R_offset
        self.ymax = top_point
        return self.xmin, self.ymin, self.xmax, self.ymax

    def draw(self, color_img, color_array=(255, 255, 0), line_weight=5, top_point=175, bot_point=675, L_offset=-135,
             R_offset=-35):
        if np.array(color_img).dtype == 'float32':
            color_img = color_img*256
        if np.array(color_img).dtype == 'uint16':
            color_img = color_img/256
        color_img = color_img.astype(np.uint8)
        for i in self.lines1:
            for x1, y1, x2, y2 in i:
                if abs((y1 - y2) / (x1 - x2 + 1E-04)) > 7:
                    cv2.line(color_img, (x1, y1), (x2, y2), (0, 255, 255), 5)
        '''draw parallelogram'''
        cv2.line(color_img,
                 ((self.Linear_X(top_point, self._B, self._K) + L_offset), top_point),
                 ((self.Linear_X(bot_point, self._B, self._K) + L_offset), bot_point),
                 color_array, line_weight)
        cv2.line(color_img,
                 ((self.Linear_X(top_point, self._B, self._K) + R_offset), top_point),
                 ((self.Linear_X(bot_point, self._B, self._K) + R_offset), bot_point),
                 color_array, line_weight)
        cv2.line(color_img,
                 ((self.Linear_X(top_point, self._B, self._K) + L_offset), top_point),
                 ((self.Linear_X(top_point, self._B, self._K) + R_offset), top_point),
                 color_array, line_weight)
        cv2.line(color_img,
                 ((self.Linear_X(bot_point, self._B, self._K) + L_offset), bot_point),
                 ((self.Linear_X(bot_point, self._B, self._K) + R_offset), bot_point),
                 color_array, line_weight)
        return color_img

    def train_search_output(self):
        self.train(), self.search(), self.output()
