import cv2
import numpy as np


class PictureProcess:

    def binirize(self, img):
        ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return bin_img

    def erode(self, img):
        kernel = np.ones((2, 2), dtype="uint8")
        img = cv2.erode(img, kernel, 3)
        return img

    def blur(self, img):
        img = cv2.GaussianBlur(img, (3, 3), 5)
        return img

    def dilatation(self, img):
        kernel = np.ones((4, 4), dtype="uint8")
        dilation = cv2.dilate(img, kernel, 3)
        return dilation

    def find_components(self, img):
        totalLabels, label_ids, values, centroid = (
            cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S))

        output = np.zeros(img.shape, dtype="uint8")
        components = 0
        areas = []
        for i in range(1, totalLabels):
            areas.append(values[i, cv2.CC_STAT_AREA])
        areas = np.array(areas)
        ind = np.argsort(areas)
        max = areas[ind][-3]
        q = np.quantile(areas, 0.5)

        for i, area in enumerate(areas):
            if q < area < max:
                components += 1
                componentMask = (label_ids == (i + 1)).astype("uint8") * 255
                output = cv2.bitwise_or(output, componentMask)

        return output, components

    def process(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output, components = self.find_components(self.binirize(self.dilatation(
                             self.blur(self.erode(img)))))
        return output, components
